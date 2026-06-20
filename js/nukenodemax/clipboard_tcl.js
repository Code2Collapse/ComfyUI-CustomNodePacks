// FILE: web/extensions/nukenodemax/clipboard_tcl.js
// FEATURE: F5 (frontend) — Ctrl+Shift+C / V intercept, server round-trip TCL
// INTEGRATES WITH: nodes/clipboard_tcl.py (/nukenodemax/copy_tcl, /nukenodemax/paste_tcl)

import { app } from "../../../scripts/app.js";

function gather(selectedNodes) {
    const idSet = new Set(selectedNodes.map(n => n.id));
    const nodes = selectedNodes.map(n => ({
        id: n.id,
        class_type: n.type,
        name: n.title || n.type,
        widgets: Object.fromEntries(
            (n.widgets || []).map(w => [w.name, w.value]).filter(([k]) => k)
        ),
        xpos: Math.round(n.pos[0]),
        ypos: Math.round(n.pos[1]),
        selected: !!n.is_selected,
    }));
    const links = [];
    for (const n of selectedNodes) {
        if (!n.inputs) continue;
        n.inputs.forEach((inp, slot) => {
            if (inp && inp.link != null) {
                const lnk = app.graph.links[inp.link];
                if (lnk && idSet.has(lnk.origin_id)) {
                    links.push([lnk.origin_id, lnk.origin_slot, n.id, slot]);
                }
            }
        });
    }
    return { nodes, links };
}

async function copyTcl() {
    const sel = Object.values(app.graph._nodes).filter(n => n.is_selected);
    if (!sel.length) {
        alert("Select one or more nodes first.");
        return;
    }
    const body = gather(sel);
    let text = "";
    try {
        const r = await fetch("/nukenodemax/copy_tcl", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        text = await r.text();
    } catch (e) {
        alert("TCL copy failed: " + e);
        return;
    }
    try {
        await navigator.clipboard.writeText(text);
    } catch {
        // Fallback: prompt the user with the text.
        prompt("Copy this TCL manually:", text);
        return;
    }
    console.info("[nukenodemax] copied", body.nodes.length, "nodes as TCL");
}

async function pasteTcl() {
    let text = "";
    try {
        text = await navigator.clipboard.readText();
    } catch {
        text = prompt("Paste TCL here:") || "";
    }
    if (!text.trim()) return;
    let parsed;
    try {
        const r = await fetch("/nukenodemax/paste_tcl", {
            method: "POST",
            headers: { "Content-Type": "text/plain" },
            body: text,
        });
        parsed = await r.json();
    } catch (e) {
        alert("TCL paste failed: " + e);
        return;
    }
    if (!parsed.ok) {
        alert("TCL parse error: " + parsed.error);
        return;
    }
    // Reconstruct nodes onto the canvas.
    const idMap = new Map();
    for (const nd of parsed.nodes) {
        const node = LiteGraph.createNode(nd.class_type);
        if (!node) {
            console.warn("[nukenodemax] unknown node class:", nd.class_type);
            continue;
        }
        node.pos = [nd.xpos, nd.ypos];
        if (nd.name) node.title = nd.name;
        for (const [k, v] of Object.entries(nd.widgets || {})) {
            const w = (node.widgets || []).find(w => w.name === k);
            if (w) w.value = v;
        }
        app.graph.add(node);
        idMap.set(nd.id, node);
    }
    for (const [fId, fSlot, tId, tSlot] of parsed.links || []) {
        const a = idMap.get(fId);
        const b = idMap.get(tId);
        if (a && b) a.connect(fSlot, b, tSlot);
    }
    app.graph.setDirtyCanvas(true, true);
    console.info("[nukenodemax] pasted", parsed.nodes.length, "nodes from TCL");
}

app.registerExtension({
    name: "nukenodemax.clipboard_tcl",
    setup() {
        document.addEventListener("keydown", (e) => {
            if (!e.ctrlKey || !e.shiftKey) return;
            if (e.key === "C" || e.key === "c") {
                e.preventDefault();
                copyTcl();
            } else if (e.key === "V" || e.key === "v") {
                e.preventDefault();
                pasteTcl();
            }
        });

        // Canvas right-click additions.
        const orig = LGraphCanvas.prototype.getCanvasMenuOptions;
        LGraphCanvas.prototype.getCanvasMenuOptions = function () {
            const opts = orig ? orig.apply(this, arguments) : [];
            opts.push(null);
            opts.push({ content: "Copy as TCL (Ctrl+Shift+C)", callback: copyTcl });
            opts.push({ content: "Paste TCL (Ctrl+Shift+V)", callback: pasteTcl });
            return opts;
        };
    },
});
