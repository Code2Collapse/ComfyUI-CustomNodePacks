/**
 * Nodes 2.0 + subgraphs: keep a C2C DOM widget showing ITS OWN element (L2.50, L2.52).
 *
 * Measured on frontend 1.53.10: "Convert to subgraph" builds a NEW node instance with the same id inside the
 * subgraph and removes the old one. In Nodes 2.0 the node's view (keyed by node id) is reused across the graph
 * switch, so it keeps showing the OLD instance's DOM widget element, a dead copy wired to a node that no longer runs,
 * while the new instance's element is never attached (the classic renderer attaches the new one). L2.52 probe over
 * every node type of the C2C packs: classic 362 / 362 fine; Nodes 2.0 313 showed a dead copy.
 *
 * Fix: the DOM widgets of every C2C pack's nodes are tagged with their widget name when the node is created (and
 * again shortly after, for widgets a node adds late). After every graph switch, and whenever a node is added to a
 * graph, each tagged widget whose element is detached replaces the stale element carrying the same tag inside its
 * node's view. Nothing happens unless that exact situation exists. Classic renderer: nothing to do. Cheap: it touches
 * only the nodes of the graph on screen, once per switch, never per frame.
 */
import { app } from "../../scripts/app.js";

const ATTR = "data-c2c-widget";
// Folder names of the C2C packs (custom_nodes.<folder>). Other authors' nodes are left alone.
const C2C_PACKS = [
    "ComfyUI-CustomNodePacks", "ComfyUI-WanNodeExperiments", "ComfyUI-NukeMaxNodes", "ComfyUI-MiniMaxSuite",
    "ComfyUI-WanAnimatePreprocessV2", "ComfyUI-LTX-Enhanced", "ComfyUI-OmniScale", "ComfyUI-GLM_Image",
    "ComfyUI-WanAnimalPreprocessor",
];

/** Mark a DOM widget so a stale copy of it can be found and replaced. */
export function tagDomWidget(widget) {
    try { if (widget?.element && widget.name) widget.element.setAttribute(ATTR, widget.name); } catch (_e) { /* ignore */ }
}

export function isC2CNode(node) {
    const mod = String(node?.constructor?.nodeData?.python_module || "");
    return C2C_PACKS.some((p) => mod === `custom_nodes.${p}` || mod.startsWith(`custom_nodes.${p}.`));
}

/** Tag every DOM widget of a C2C node that is not tagged yet. */
export function tagNode(node) {
    if (!node || !isC2CNode(node)) return 0;
    let n = 0;
    for (const w of node.widgets || []) {
        if (w?.element && w.name && !w.element.getAttribute?.(ATTR)) { tagDomWidget(w); n += 1; }
    }
    return n;
}

/** Put the node's own tagged widget elements back into its Nodes 2.0 view. Returns how many were swapped. */
export function rebindNode(node) {
    if (!globalThis.LiteGraph?.vueNodesMode || !node) return 0;
    let host = null;
    try { host = document.querySelector(`[data-node-id="${CSS.escape(String(node.id))}"]`); } catch (_e) { return 0; }
    if (!host) return 0;
    let swapped = 0;
    for (const w of node.widgets || []) {
        const el = w?.element;
        if (!el || el.isConnected || !el.getAttribute?.(ATTR)) continue;
        const stale = host.querySelector(`[${ATTR}="${CSS.escape(String(w.name))}"]`);
        if (stale && stale !== el) {
            stale.replaceWith(el);
            swapped += 1;
        }
    }
    return swapped;
}

export function rebindGraph(graph) {
    let n = 0;
    for (const node of graph?._nodes || graph?.nodes || []) {
        tagNode(node);
        n += rebindNode(node);
    }
    return n;
}

/** After Vue has rendered: two frames, then a short timeout for lazily mounted widgets. */
export function afterRender(fn) {
    requestAnimationFrame(() => requestAnimationFrame(() => { fn(); setTimeout(fn, 120); }));
}

if (!globalThis.__c2cVueDomRebind) {
    globalThis.__c2cVueDomRebind = true;
    app.registerExtension({
        name: "C2C.VueDomRebind",
        nodeCreated(node) {
            if (!isC2CNode(node)) return;
            tagNode(node);
            // a panel added at any later time (Folder Incrementer adds its status strip after a server probe) is
            // tagged the moment it exists: wrap THIS node's addDOMWidget, never the prototype
            const add = node.addDOMWidget;
            if (typeof add === "function" && !add.__c2cTagging) {
                const wrapped = function (...args) {
                    const w = add.apply(this, args);
                    tagDomWidget(w);
                    return w;
                };
                wrapped.__c2cTagging = true;
                node.addDOMWidget = wrapped;
            }
        },
        setup() {
            const canvasEl = app.canvas?.canvas;
            canvasEl?.addEventListener?.("litegraph:set-graph", (e) => {
                const g = e?.detail?.newGraph || app.canvas?.graph;
                afterRender(() => rebindGraph(g));
            });
        },
    });
}
