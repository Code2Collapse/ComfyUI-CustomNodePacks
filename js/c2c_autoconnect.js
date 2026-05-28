// c2c_autoconnect.js — Slot auto-connect & session learning.
//
// Provides:
//   * Double-click an unconnected slot dot  → spawn top-1 predicted node + link
//     (Shift+double-click = greedy chain).
//   * Programmatic API for the popover "Insert" button:
//       window.c2c_autoconnect.insertSuggestion({...})
//   * Session tracker: every connect that happens in the UI is POSTed to
//     /c2c/autoconnect/record so the predictor keeps learning.
//
// Frontend file only — backend lives in nodes/_c2c_autoconnect.py.

import { app } from "../../../scripts/app.js";

const TAG = "[C2C/autoconnect]";
const SLOT_RADIUS = 10;      // graph-px tolerance for slot dot hit-tests
const CHAIN_MAX   = 8;       // safety cap on Shift+double-click chains
const CHAIN_MIN_CONF = 0.35; // stop chain when top suggestion is weaker than this
const NEARBY_RADIUS = 600;   // graph-px search radius for existing-node match on dblclick

// ---------------------------------------------------------------------------
// Geometry helpers (kept independent of c2c_node_explain so this file works
// standalone in case the explain panel is disabled).
// ---------------------------------------------------------------------------
function _eventToGraph(e) {
    // LiteGraph transform: screen = (graph + offset) * scale
    // Inverse:             graph  =  screen/scale - offset
    const canvas = app.canvas;
    const rect = canvas.canvas.getBoundingClientRect();
    const ds = canvas.ds;
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    return {
        gx: cx / ds.scale - ds.offset[0],
        gy: cy / ds.scale - ds.offset[1],
    };
}

function _slotHitTest(gx, gy) {
    const graph = app.canvas?.graph;
    if (!graph || !Array.isArray(graph._nodes)) return null;
    const nodes = graph._nodes;
    const R = SLOT_RADIUS;
    for (let i = nodes.length - 1; i >= 0; i--) {
        const n = nodes[i];
        if (!n || n.flags?.collapsed) continue;
        const [nx, ny] = n.pos || [0, 0];
        const [nw, nh] = n.size || [0, 0];
        // AABB reject with slot radius margin
        if (gx < nx - R || gx > nx + nw + R || gy < ny - R || gy > ny + nh + R) continue;
        const probe = (slots, isInput) => {
            if (!Array.isArray(slots)) return null;
            for (let j = 0; j < slots.length; j++) {
                let p;
                try { p = n.getConnectionPos(isInput, j); }
                catch (_e) { p = [isInput ? nx : nx + nw, ny + 20 * (j + 0.5)]; }
                const dx = gx - p[0], dy = gy - p[1];
                if (dx * dx + dy * dy <= R * R) return { node: n, isInput, slotIndex: j };
            }
            return null;
        };
        const hitIn = probe(n.inputs, true);
        if (hitIn) return hitIn;
        const hitOut = probe(n.outputs, false);
        if (hitOut) return hitOut;
    }
    return null;
}

// ---------------------------------------------------------------------------
// Nearby existing-node lookup.
//   Goal: when the user double-clicks a free slot, FIRST try to reuse an
//   existing compatible node within `NEARBY_RADIUS` graph-px instead of
//   always spawning a new one. Scoring favours, in order:
//     1) Exact class match with the predicted class                   (200)
//     2) A free slot of the exact same type as the anchor slot        (100)
//     3) A free wildcard ("*") slot                                   ( 50)
//   Then subtracts the squared distance / 2000 so closer wins.
//   Returns { node, slotIndex } or null.
//
//   `wantInputOnTarget` is the direction we need on the candidate node:
//   if the anchor is an OUTPUT we look for an INPUT on the candidate; if
//   the anchor is an INPUT we look for an OUTPUT.
// ---------------------------------------------------------------------------
function _isTypeCompatible(slotType, wanted) {
    if (slotType === -1 || slotType === undefined || slotType === null) return "wild";
    if (slotType === "*" || slotType === 0) return "wild";
    if (Array.isArray(slotType)) {
        if (slotType.includes(wanted) || slotType.includes("*")) return "exact";
        return false;
    }
    if (String(slotType) === String(wanted)) return "exact";
    return false;
}

function _findNearbyCompatibleNode(anchorNode, anchorIsInput, anchorType, preferredCls) {
    const graph = app.canvas?.graph || app.graph;
    if (!graph || !Array.isArray(graph._nodes)) return null;
    const wantInputOnTarget = !anchorIsInput;
    const [ax, ay] = anchorNode.pos || [0, 0];
    const [aw, ah] = anchorNode.size || [200, 100];
    // Anchor reference point: outgoing right edge or incoming left edge.
    const refX = anchorIsInput ? ax : ax + aw;
    const refY = ay + ah / 2;
    const R2 = NEARBY_RADIUS * NEARBY_RADIUS;

    let best = null;     // {node, slotIndex, score}
    let bestScore = -Infinity;

    const nodes = graph._nodes;
    for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        if (!n || n === anchorNode) continue;
        if (n.flags?.collapsed) continue;
        const [nx, ny] = n.pos || [0, 0];
        const [nw, nh] = n.size || [200, 100];
        const cx = nx + nw / 2;
        const cy = ny + nh / 2;
        const d2 = (cx - refX) * (cx - refX) + (cy - refY) * (cy - refY);
        if (d2 > R2) continue;

        // Side gate: prefer right of anchor for output→input, left for input→output.
        // Soft gate — we don't outright reject, just penalize wrong side.
        const wrongSide = anchorIsInput ? (nx > ax) : (nx + nw < ax);

        const slots = wantInputOnTarget ? (n.inputs || []) : (n.outputs || []);
        for (let j = 0; j < slots.length; j++) {
            const s = slots[j];
            if (!s) continue;
            // For inputs: skip if already linked; for outputs: still allow (fan-out is fine).
            if (wantInputOnTarget && s.link != null) continue;
            const compat = _isTypeCompatible(s.type, anchorType);
            if (!compat) continue;
            let score = (compat === "exact") ? 100 : 50;
            if (preferredCls && n.type === preferredCls) score += 200;
            score -= d2 / 2000;
            if (wrongSide) score -= 80;
            if (score > bestScore) {
                bestScore = score;
                best = { node: n, slotIndex: j, score };
            }
        }
    }
    return best;
}

function _connectExisting(anchorNode, anchorIsInput, anchorSlotIndex, targetNode, targetSlotIndex) {
    try {
        let ok;
        if (anchorIsInput) {
            // target.output → anchor.input
            ok = targetNode.connect(targetSlotIndex, anchorNode, anchorSlotIndex) !== null;
        } else {
            // anchor.output → target.input
            ok = anchorNode.connect(anchorSlotIndex, targetNode, targetSlotIndex) !== null;
        }
        (app.canvas?.graph || app.graph)?.setDirtyCanvas?.(true, true);
        return ok;
    } catch (e) {
        console.warn(TAG, "connect-existing failed:", e);
        return false;
    }
}

// ---------------------------------------------------------------------------
// Suggestion fetch
// ---------------------------------------------------------------------------
async function _fetchSuggestions(node, isInput, slotIndex, limit = 5) {
    const slot = (isInput ? node.inputs : node.outputs)?.[slotIndex];
    if (!slot) return [];
    const cls = node.type || "";
    const dir = isInput ? "input" : "output";
    const slotName = slot.name || "";
    let slotType = "*";
    if (slot.type !== undefined && slot.type !== null && slot.type !== -1) {
        slotType = Array.isArray(slot.type) ? String(slot.type[0]) : String(slot.type);
    }
    try {
        const url = `/c2c/autoconnect/suggest?cls=${encodeURIComponent(cls)}&dir=${dir}&slot=${encodeURIComponent(slotName)}&type=${encodeURIComponent(slotType)}&limit=${limit|0}`;
        const r = await fetch(url);
        if (!r.ok) return [];
        const j = await r.json();
        return Array.isArray(j?.suggestions) ? j.suggestions : [];
    } catch (_e) {
        return [];
    }
}

// ---------------------------------------------------------------------------
// Slot resolution on the spawned node — pick the input/output slot by NAME,
// or fall back to the first compatible-type slot.
// ---------------------------------------------------------------------------
function _resolveSlotIndex(node, byName, wantInput, fallbackType) {
    const slots = wantInput ? (node.inputs || []) : (node.outputs || []);
    if (byName) {
        const idx = slots.findIndex(s => s && s.name === byName);
        if (idx >= 0) return idx;
    }
    if (fallbackType) {
        const idx = slots.findIndex(s => {
            if (!s) return false;
            if (s.type === -1 || s.type === undefined || s.type === null || s.type === "*") return true;
            if (Array.isArray(s.type)) return s.type.includes(fallbackType);
            return String(s.type) === String(fallbackType);
        });
        if (idx >= 0) return idx;
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Spawn + connect a predicted node next to its anchor.
// ---------------------------------------------------------------------------
function _spawnAndConnect(anchorNode, anchorIsInput, anchorSlotIndex, prediction) {
    const { LiteGraph } = window;
    if (!LiteGraph || !LiteGraph.createNode) {
        console.warn(TAG, "LiteGraph unavailable");
        return null;
    }
    const dstCls = prediction.cls;
    const dstSlotName = prediction.slot || "";
    const node = LiteGraph.createNode(dstCls);
    if (!node) {
        console.warn(TAG, "unknown class:", dstCls);
        return null;
    }
    // Determine the source slot's type so we can fall back to first-of-type
    // when the predicted slot name doesn't exist on the target node.
    const anchorSlot = (anchorIsInput ? anchorNode.inputs : anchorNode.outputs)?.[anchorSlotIndex];
    let anchorType = "*";
    if (anchorSlot?.type !== undefined && anchorSlot.type !== null && anchorSlot.type !== -1) {
        anchorType = Array.isArray(anchorSlot.type) ? String(anchorSlot.type[0]) : String(anchorSlot.type);
    }

    // Position: place to the right of source (if anchor is output) or left
    // (if anchor is input).  Tries to avoid overlap by stepping 50px down for
    // every existing node that intersects the target rect.
    const [ax, ay] = anchorNode.pos || [0, 0];
    const [aw, ah] = anchorNode.size || [200, 100];
    let nx = anchorIsInput ? (ax - 280) : (ax + aw + 60);
    let ny = ay;
    const graph = app.canvas?.graph || app.graph;
    if (graph?._nodes?.length) {
        const overlap = (x, y) => graph._nodes.some(n => {
            if (!n || n === anchorNode) return false;
            const [px, py] = n.pos || [0, 0];
            const [pw, ph] = n.size || [200, 100];
            return Math.abs(px - x) < 220 && Math.abs(py - y) < 80;
        });
        let guard = 0;
        while (overlap(nx, ny) && guard < 40) { ny += 70; guard++; }
    }
    node.pos = [nx, ny];
    graph.add(node);

    // Resolve the matching slot on the new node (opposite direction of anchor).
    const newWantInput = !anchorIsInput;
    const newSlotIndex = _resolveSlotIndex(node, dstSlotName, newWantInput, anchorType);
    if (newSlotIndex < 0) {
        console.warn(TAG, "couldn't resolve slot", dstSlotName, "on", dstCls);
        graph.setDirtyCanvas(true, true);
        return { node, ok: false };
    }

    // Make the link.
    let ok = false;
    try {
        if (anchorIsInput) {
            ok = node.connect(newSlotIndex, anchorNode, anchorSlotIndex) !== null;
        } else {
            ok = anchorNode.connect(anchorSlotIndex, node, newSlotIndex) !== null;
        }
    } catch (e) {
        console.warn(TAG, "connect failed:", e);
    }
    graph.setDirtyCanvas(true, true);
    return { node, ok, newSlotIndex };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
async function insertSuggestion({ sourceNodeId, sourceSlotIndex, sourceIsInput, destClass, destSlotName }) {
    const graph = app.canvas?.graph || app.graph;
    const src = graph?.getNodeById?.(sourceNodeId);
    if (!src) return null;
    const res = _spawnAndConnect(src, sourceIsInput, sourceSlotIndex, { cls: destClass, slot: destSlotName });
    return res;
}

async function autoconnectFromSlot(node, isInput, slotIndex, opts = {}) {
    const { chain = false } = opts;
    const slot = (isInput ? node.inputs : node.outputs)?.[slotIndex];
    if (!slot) return;
    // Refuse if the slot is already connected (avoid silent fan-out spam on inputs;
    // outputs can still chain since they can fan out).
    if (isInput && slot.link != null) {
        console.info(TAG, "input already connected — skipping autoconnect");
        return;
    }
    // Resolve anchor type once — used by both the nearby-node lookup and the
    // server suggestion fallback.
    let anchorType = "*";
    if (slot.type !== undefined && slot.type !== null && slot.type !== -1) {
        anchorType = Array.isArray(slot.type) ? String(slot.type[0]) : String(slot.type);
    }

    // Step 1 — Ask the predictor for the *preferred class* so the nearby
    // search can prefer "VAEDecode" (or whatever the model expects) over
    // an arbitrary type-compatible node. This call is async but small.
    const sugg = await _fetchSuggestions(node, isInput, slotIndex, 5);
    const top = sugg[0];
    const preferredCls = top?.cls || null;

    // Step 2 — Try to reuse a nearby existing compatible node FIRST. This is
    // what users actually expect: if a `VAEDecode` is already sitting on the
    // canvas next to the slot, dblclick should just connect to it, not spawn
    // a second one.
    const reuse = _findNearbyCompatibleNode(node, isInput, anchorType, preferredCls);
    if (reuse) {
        const ok = _connectExisting(node, isInput, slotIndex, reuse.node, reuse.slotIndex);
        if (ok) {
            try {
                app.extensionManager?.toast?.add?.({
                    severity: "info",
                    summary: "Auto-connected",
                    detail: `Linked to existing ${reuse.node.title || reuse.node.type}`,
                    life: 1800,
                });
            } catch (_) {}
            return;
        }
        // If the connect failed for some reason, fall through to spawn path.
    }

    // Step 3 — No nearby reuse → spawn a fresh node (the original behaviour).
    if (!sugg.length) {
        console.info(TAG, "no suggestions for", node.type, slot.name);
        return;
    }
    const res = _spawnAndConnect(node, isInput, slotIndex, top);
    if (!res?.ok || !chain) return;

    // Chain: from the newly-spawned node's OUTPUT (slot 0 by convention) recurse.
    let cur = res.node;
    let curOutIndex = 0;
    for (let step = 0; step < CHAIN_MAX; step++) {
        if (!cur.outputs || cur.outputs.length === 0) break;
        // Prefer first output that has no link yet
        curOutIndex = cur.outputs.findIndex(o => !o?.links || o.links.length === 0);
        if (curOutIndex < 0) curOutIndex = 0;
        const next = await _fetchSuggestions(cur, false, curOutIndex, 3);
        if (!next.length) break;
        const pick = next[0];
        if ((pick.confidence || 0) < CHAIN_MIN_CONF) break;
        const r = _spawnAndConnect(cur, false, curOutIndex, pick);
        if (!r?.ok) break;
        cur = r.node;
    }
}

// ---------------------------------------------------------------------------
// Session learning: hook LGraphNode.prototype.connect so every successful
// connection is POSTed to /c2c/autoconnect/record.
// ---------------------------------------------------------------------------
function _hookConnect() {
    const proto = window.LiteGraph?.LGraphNode?.prototype;
    if (!proto || proto.__c2c_ac_hooked) return;
    const orig = proto.connect;
    proto.connect = function patched(srcSlotIdx, target, dstSlotIdx) {
        const ret = orig.apply(this, arguments);
        try {
            if (ret !== null && ret !== false && target && typeof target === "object") {
                const sCls = this.type || "";
                const sSlot = this.outputs?.[srcSlotIdx]?.name || "";
                const dCls = target.type || "";
                let dIdx = dstSlotIdx;
                if (typeof dIdx === "string") {
                    // ComfyUI sometimes passes the slot NAME — resolve to index.
                    dIdx = (target.inputs || []).findIndex(s => s?.name === dstSlotIdx);
                }
                const dSlot = target.inputs?.[dIdx]?.name || (typeof dstSlotIdx === "string" ? dstSlotIdx : "");
                if (sCls && dCls) {
                    fetch("/c2c/autoconnect/record", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ edges: [[sCls, sSlot, dCls, dSlot]] }),
                    }).catch(() => {});
                }
            }
        } catch (_e) { /* never break LiteGraph */ }
        return ret;
    };
    proto.__c2c_ac_hooked = true;
}

// ---------------------------------------------------------------------------
// Double-click hit-detection on the canvas
// ---------------------------------------------------------------------------
function _attachDoubleClickHandler() {
    const canvasEl = app.canvas?.canvas;
    if (!canvasEl || canvasEl.__c2c_ac_dblclick) return;
    canvasEl.addEventListener("dblclick", (e) => {
        try {
            const { gx, gy } = _eventToGraph(e);
            const hit = _slotHitTest(gx, gy);
            if (!hit) return;
            // Skip if already connected & it's an input slot (avoid surprise replacement)
            const slot = (hit.isInput ? hit.node.inputs : hit.node.outputs)?.[hit.slotIndex];
            if (!slot) return;
            if (hit.isInput && slot.link != null) return;
            // Stop LiteGraph's own double-click handler (which spawns Reroute / Get/Set)
            e.preventDefault();
            e.stopPropagation();
            const chain = !!e.shiftKey;
            autoconnectFromSlot(hit.node, hit.isInput, hit.slotIndex, { chain });
        } catch (err) {
            console.warn(TAG, "dblclick handler failed:", err);
        }
    }, true);  // capture so we run before LiteGraph
    canvasEl.__c2c_ac_dblclick = true;
}

// ---------------------------------------------------------------------------
// Extension registration
// ---------------------------------------------------------------------------
app.registerExtension({
    name: "C2C.Autoconnect",
    settings: [
        {
            id: "c2c.autoconnect.enabled",
            name: "Auto-connect: double-click slot to insert predicted node",
            type: "boolean",
            defaultValue: true,
        },
    ],
    async setup() {
        // expose API for popover "Insert" buttons
        window.c2c_autoconnect = Object.assign(window.c2c_autoconnect || {}, {
            insertSuggestion,
            autoconnectFromSlot,
            fetchSuggestions: _fetchSuggestions,
            slotHitTest: _slotHitTest,
            version: 1,
        });

        const enabled = () => {
            try {
                return app.ui?.settings?.getSettingValue?.("c2c.autoconnect.enabled", true) ?? true;
            } catch (_e) { return true; }
        };

        // Wait for canvas, then attach
        let attempts = 0;
        const tryAttach = () => {
            if (app.canvas?.canvas) {
                if (enabled()) {
                    _attachDoubleClickHandler();
                    _hookConnect();
                    console.info(TAG, "ready (double-click + session tracker installed)");
                }
            } else if (attempts++ < 60) {
                setTimeout(tryAttach, 250);
            }
        };
        tryAttach();
    },
});
