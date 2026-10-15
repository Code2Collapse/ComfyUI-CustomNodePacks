/**
 * Image batching diagnostics (L2.54 slice 2; owner A14: "Badge + sidebar").
 *
 * The server sends `c2c.autobatch.decision` when Image batching changed how a node ran (a proven-faster operation
 * was used, or the batch was split into chunks), tagged with the node id core was running. This module:
 *   - puts a small badge on that node (core's own node badges: the classic title bar and the Nodes 2.0 header both
 *     draw `node.badges`), cleared when the next run starts;
 *   - keeps the last run's decisions for the "Batching" tab of the C2C Diagnostics sidebar.
 * Costs nothing between runs: no polling, it only reacts to the server's events.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const RUN = [];                 // decisions of the current / last run, in order
const MARK = "__c2cBatchBadge";
let tableHost = null;

function findNode(id) {
    const key = String(id ?? "");
    if (!key) return null;
    const parts = key.split(":");
    let graph = app.graph;
    let node = null;
    for (const p of parts) {
        node = graph?.getNodeById?.(Number(p)) ?? graph?.getNodeById?.(p) ?? null;
        if (!node) return null;
        graph = node.subgraph || graph;
    }
    return node;
}

function badgeText(e) {
    if (e.strategy === "adapter") {
        const x = Number(e.speedup_predicted);
        return Number.isFinite(x) && x > 0 ? `⚡ ${x.toFixed(1)}×` : "⚡ faster";
    }
    if (e.strategy === "chunked") return `▦ ${e.chunks}×${e.chunk_size}`;
    return null;
}

function clearBadges(graph = app.graph, seen = new Set()) {
    if (!seen.size) document.querySelectorAll(`.${PILL}`).forEach((el) => el.remove());
    if (!graph || seen.has(graph)) return;
    seen.add(graph);
    for (const n of graph._nodes || graph.nodes || []) {
        if (Array.isArray(n.badges) && n.badges.some((b) => b?.[MARK])) n.badges = n.badges.filter((b) => !b?.[MARK]);
        if (n.subgraph) clearBadges(n.subgraph, seen);
    }
}

const PILL = "c2c-batch-pill";

function vuePill(node, text, bg, title) {
    // Nodes 2.0 builds its header from node.badges when the node view mounts and does not re-read it for a node
    // that is already on screen (measured, frontend 1.53.10), so the pill goes into the header element itself.
    let host = null;
    try { host = document.querySelector(`[data-node-id="${CSS.escape(String(node.id))}"] .lg-node-header`); } catch (_e) { /* ignore */ }
    if (!host) return;
    host.querySelector(`.${PILL}`)?.remove();
    const s = document.createElement("span");
    s.className = PILL;
    s.textContent = text;
    s.title = title;
    s.style.cssText = `margin-left:6px;padding:1px 6px;border-radius:5px;font-size:11px;line-height:16px;font-weight:600;`
        + `color:#0c0d23;background:${bg};white-space:nowrap;flex:none;`;
    host.appendChild(s);
}

function setBadge(e) {
    const text = badgeText(e);
    const node = findNode(e.node);
    if (!text || !node) return;
    const bg = e.strategy === "adapter" ? "#8fd18f" : "#f5c26b";
    const Badge = globalThis.LGraphBadge;
    if (typeof Badge === "function") {
        const b = new Badge({ text, fgColor: "#0c0d23", bgColor: bg, fontSize: 11 });
        b[MARK] = true;
        node.badges = [...(node.badges || []).filter((x) => !x?.[MARK]), b];
    }
    if (globalThis.LiteGraph?.vueNodesMode) vuePill(node, text, bg, fmt(e));
    node.setDirtyCanvas?.(true, true);
}

function fmt(e) {
    if (e.strategy === "adapter") return `faster version · ${e.seconds ?? "?"} s · predicted ${e.speedup_predicted ?? "?"}×`;
    if (e.strategy === "chunked") return `split into ${e.chunks} chunks of ${e.chunk_size} frames`;
    return e.reason || "original";
}

/** The sidebar tab body: the last run's decisions. */
export function renderBatchingTable(el) {
    tableHost = el;
    el.innerHTML = "";
    const box = document.createElement("div");
    box.style.cssText = "display:flex;flex-direction:column;gap:6px;padding:8px;font-size:12px;";
    const head = document.createElement("div");
    head.style.cssText = "opacity:.8;line-height:1.4;";
    head.textContent = RUN.length
        ? "Nodes where Image batching changed how they ran in the last run (Settings > C2C > Image batching)."
        : "Run a workflow. Nodes where Image batching changed how they ran show up here and get a badge on the canvas.";
    box.appendChild(head);
    if (RUN.length) {
        const table = document.createElement("table");
        table.style.cssText = "border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;";
        for (const e of RUN) {
            const tr = document.createElement("tr");
            for (const v of [String(e.node ?? ""), String(e.label ?? ""), String(e.frames ?? ""), fmt(e)]) {
                const td = document.createElement("td");
                td.textContent = v;
                td.style.cssText = "padding:3px 6px;border-bottom:1px solid rgba(127,127,127,.25);vertical-align:top;";
                tr.appendChild(td);
            }
            tr.style.cursor = "pointer";
            tr.title = "Select this node";
            tr.onclick = () => {
                const n = findNode(e.node);
                if (n) { app.canvas.selectNode?.(n); app.canvas.centerOnNode?.(n); }
            };
            table.appendChild(tr);
        }
        box.appendChild(table);
    }
    el.appendChild(box);
}

if (!globalThis.__c2cBatchingBadges) {
    globalThis.__c2cBatchingBadges = true;
    app.registerExtension({
        name: "C2C.BatchingBadges",
        setup() {
            api.addEventListener("execution_start", () => {
                RUN.length = 0;
                clearBadges();
                if (tableHost?.isConnected) renderBatchingTable(tableHost);
            });
            api.addEventListener("c2c.autobatch.decision", (ev) => {
                const e = ev?.detail;
                if (!e || e.strategy === "original") return;
                RUN.push(e);
                setBadge(e);
                if (tableHost?.isConnected) renderBatchingTable(tableHost);
            });
        },
    });
}
