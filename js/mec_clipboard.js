// MEC clipboard — Nuke-style portable copy/paste for ComfyUI nodes.
//
// Ports the spirit of NukeMax's NkScript clipboard, but is fully
// self-contained: no server endpoint, no TCL grammar. The payload is
// plain JSON wrapped in a recognisable header so it round-trips through
// the OS clipboard between workflows / instances / machines.
//
// Hotkeys (chosen to avoid clashing with NukeMax's Ctrl+Shift+C/V and
// LiteGraph's own Ctrl+C/V):
//   Ctrl+Alt+C   → copy selected nodes (all widget values, links,
//                   relative positions, sizes, colours) to the clipboard.
//   Ctrl+Alt+V   → paste the clipboard payload at the cursor.
//
// The payload includes EVERY widget value by name, so even if the user
// pastes into a different workflow / different ComfyUI install (as long
// as the same node packs are installed), the settings come across.

import { app } from "../../scripts/app.js";

const CLIP_HEADER = "# MEC.clipboard 1.0";

function _toast(msg, severity = "info") {
    try {
        app.extensionManager?.toast?.add({
            severity,
            summary: "MEC Clipboard",
            detail: msg,
            life: 3500,
        });
    } catch (_) { console.log("[MEC.clipboard]", msg); }
}

function _selectedNodes() {
    const sel = app.canvas?.selected_nodes;
    if (!sel) return [];
    return Object.values(sel);
}

function _serializeNode(n) {
    // Capture every widget value by NAME (so widget-order changes don't
    // break the paste).
    const widgets = {};
    for (const w of n.widgets || []) {
        if (!w.name || w.value === undefined) continue;
        // Skip DOM-widget values that aren't JSON-serialisable (canvas
        // payloads, typed arrays). serializeValue() if available.
        let v = w.value;
        try {
            if (typeof w.serializeValue === "function") {
                v = w.serializeValue(n, n.widgets.indexOf(w));
            }
            JSON.stringify(v);
        } catch (_) { continue; }
        widgets[w.name] = v;
    }
    return {
        id: n.id,
        type: n.comfyClass || n.type,
        title: n.title,
        pos: [Math.round(n.pos?.[0] ?? 0), Math.round(n.pos?.[1] ?? 0)],
        size: n.size ? [Math.round(n.size[0]), Math.round(n.size[1])] : undefined,
        color: n.color,
        bgcolor: n.bgcolor,
        flags: n.flags,
        properties: n.properties,
        widgets,
    };
}

function _gatherSubgraph(nodes) {
    const ids = new Set(nodes.map((n) => n.id));
    const out_nodes = nodes.map(_serializeNode);
    const out_links = [];
    const links = app.graph?.links || {};
    for (const k in links) {
        const l = links[k];
        if (!l) continue;
        if (ids.has(l.origin_id) && ids.has(l.target_id)) {
            out_links.push({
                src_id: l.origin_id, src_slot: l.origin_slot,
                dst_id: l.target_id, dst_slot: l.target_slot,
                type: l.type,
            });
        }
    }
    return { version: 1, nodes: out_nodes, links: out_links };
}

async function _copy() {
    const nodes = _selectedNodes();
    if (!nodes.length) { _toast("Nothing selected", "warn"); return; }
    const payload = _gatherSubgraph(nodes);
    const text = CLIP_HEADER + "\n" + JSON.stringify(payload, null, 2);
    try {
        await navigator.clipboard.writeText(text);
        _toast(`Copied ${nodes.length} node(s) with full metadata`, "success");
    } catch (e) {
        _toast("Clipboard write denied: " + (e?.message || e), "error");
    }
}

function _findHeader(text) {
    if (!text) return null;
    const i = text.indexOf(CLIP_HEADER);
    if (i < 0) return null;
    const j = text.indexOf("{", i);
    if (j < 0) return null;
    try { return JSON.parse(text.slice(j)); } catch (_) { return null; }
}

async function _paste() {
    let text = "";
    try { text = await navigator.clipboard.readText(); }
    catch (e) { _toast("Clipboard read denied (browser permission)", "error"); return; }
    const data = _findHeader(text);
    if (!data || !Array.isArray(data.nodes)) {
        _toast("Clipboard does not contain MEC payload", "warn"); return;
    }
    const cursor = app.canvas?.graph_mouse || [0, 0];
    let originX = null, originY = null;
    const idMap = {}; // old_id -> new node
    for (const nd of data.nodes) {
        const node = LiteGraph.createNode(nd.type);
        if (!node) {
            console.warn("[MEC.clipboard] unknown node type:", nd.type);
            _toast(`Unknown node type "${nd.type}" — install the pack`, "warn");
            continue;
        }
        if (originX === null) { originX = nd.pos[0]; originY = nd.pos[1]; }
        node.pos = [
            cursor[0] + (nd.pos[0] - originX),
            cursor[1] + (nd.pos[1] - originY),
        ];
        if (nd.title) node.title = nd.title;
        if (nd.color) node.color = nd.color;
        if (nd.bgcolor) node.bgcolor = nd.bgcolor;
        if (nd.flags) node.flags = { ...(node.flags || {}), ...nd.flags };
        if (nd.properties) node.properties = { ...(node.properties || {}), ...nd.properties };
        app.graph.add(node);
        if (Array.isArray(nd.size)) {
            // Apply AFTER add() so LiteGraph doesn't auto-resize on top of us.
            node.size = [nd.size[0], nd.size[1]];
        }
        // Apply widgets by NAME.
        for (const [wname, wval] of Object.entries(nd.widgets || {})) {
            const w = (node.widgets || []).find((x) => x.name === wname);
            if (!w) continue;
            try {
                w.value = wval;
                w.callback?.(wval, app.canvas, node);
            } catch (err) {
                console.warn("[MEC.clipboard] widget set failed:", wname, err);
            }
        }
        idMap[nd.id] = node;
    }
    // Re-wire links by old IDs.
    for (const l of data.links || []) {
        const src = idMap[l.src_id], dst = idMap[l.dst_id];
        if (!src || !dst) continue;
        try { src.connect(l.src_slot, dst, l.dst_slot); }
        catch (err) { console.warn("[MEC.clipboard] link failed", err); }
    }
    // Select the freshly pasted nodes.
    try {
        app.canvas.deselectAllNodes?.();
        for (const n of Object.values(idMap)) app.canvas.selectNode?.(n, true);
    } catch (_) {}
    app.graph.setDirtyCanvas(true, true);
    _toast(`Pasted ${Object.keys(idMap).length} node(s)`, "success");
}

window.addEventListener("keydown", (ev) => {
    if (!ev.ctrlKey || !ev.altKey || ev.shiftKey) return;
    // Ignore if focus is in a text input — let the user paste into fields.
    const t = ev.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
    const k = ev.key.toLowerCase();
    if (k === "c") { ev.preventDefault(); _copy(); }
    else if (k === "v") { ev.preventDefault(); _paste(); }
});

app.registerExtension({
    name: "MEC.Clipboard",
    async setup() {
        console.log("[MEC.clipboard] portable JSON copy/paste loaded (Ctrl+Alt+C / Ctrl+Alt+V)");
        const origMenu = LGraphCanvas.prototype.getCanvasMenuOptions;
        LGraphCanvas.prototype.getCanvasMenuOptions = function () {
            const opts = origMenu.apply(this, arguments) || [];
            opts.push(null);
            opts.push({ content: "MEC: Copy node(s) with metadata (Ctrl+Alt+C)", callback: _copy });
            opts.push({ content: "MEC: Paste node(s) from clipboard (Ctrl+Alt+V)", callback: _paste });
            return opts;
        };
        // Also expose on per-node right-click menu.
        const origNodeMenu = LGraphCanvas.prototype.getNodeMenuOptions;
        LGraphCanvas.prototype.getNodeMenuOptions = function (node) {
            const opts = origNodeMenu.apply(this, arguments) || [];
            opts.push(null);
            opts.push({ content: "MEC: Copy with metadata", callback: () => {
                // Ensure this node is in the selection.
                try { app.canvas.selectNode?.(node, true); } catch (_) {}
                _copy();
            }});
            return opts;
        };
    },
});
