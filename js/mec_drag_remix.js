/**
 * mec_drag_remix.js — Phase 17d: Drag-to-Remix
 *
 * Drop an image file from disk or a session output thumbnail onto the
 * canvas and a LoadImage node is automatically spawned at the drop point
 * pre-filled with the uploaded filename. Optionally also wires it into the
 * nearest VAEEncode latent input found upstream of any KSampler.
 *
 * Setting:
 *   mec.drag_remix.enabled — bool (default true)
 *   mec.drag_remix.auto_wire — bool (default false)
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

async function _uploadImage(file) {
    const fd = new FormData();
    fd.append("image", file, file.name);
    fd.append("overwrite", "false");
    const resp = await api.fetchApi("/upload/image", { method: "POST", body: fd });
    if (!resp.ok) throw new Error(`upload failed: ${resp.status}`);
    return await resp.json();  // { name, subfolder, type }
}

function _canvasPosFromEvent(ev) {
    const canvas = app.canvas;
    const rect = canvas.canvas.getBoundingClientRect();
    const sx = (ev.clientX - rect.left);
    const sy = (ev.clientY - rect.top);
    const scale = canvas.ds.scale;
    const ox = canvas.ds.offset[0];
    const oy = canvas.ds.offset[1];
    return [sx / scale - ox, sy / scale - oy];
}

function _spawnLoadImage(filename, pos) {
    if (!window.LiteGraph) return null;
    const node = LiteGraph.createNode("LoadImage");
    if (!node) {
        console.warn("[MEC.DragRemix] LoadImage not registered");
        return null;
    }
    app.graph.add(node);
    node.pos = [pos[0] - 100, pos[1] - 30];
    // The LoadImage widget is a combo of filenames; set value directly.
    const w = (node.widgets || []).find(x => x && x.name === "image");
    if (w) {
        w.value = filename;
        if (typeof w.callback === "function") {
            try { w.callback(filename, app.canvas, node); } catch { /* ignore */ }
        }
    }
    app.canvas.setDirty(true, true);
    return node;
}

function _findKSampler() {
    const g = app.graph;
    if (!g || !g._nodes) return null;
    return g._nodes.find(n => /ksampler/i.test(n.type || ""));
}

function _findVAEEncodeFeedingSampler(sampler) {
    // Look at sampler latent_image input → trace upstream until we hit a VAEEncode node.
    if (!sampler) return null;
    const g = app.graph;
    const targetSlotIdx = (sampler.inputs || []).findIndex(i => /latent_image/i.test(i?.name || ""));
    if (targetSlotIdx < 0) return null;
    const link = sampler.inputs[targetSlotIdx]?.link;
    if (!link) return null;
    const linkInfo = g.links?.[link];
    const upstream = linkInfo && g.getNodeById(linkInfo.origin_id);
    if (!upstream) return null;
    if (/vaeencode/i.test(upstream.type || "")) return upstream;
    return null;
}

function _autoWire(loadNode) {
    const enabled = (() => {
        try { return app.ui.settings.getSettingValue("mec.drag_remix.auto_wire", false); }
        catch { return false; }
    })();
    if (!enabled || !loadNode) return;
    const sampler = _findKSampler();
    const vaeEncode = _findVAEEncodeFeedingSampler(sampler);
    if (!vaeEncode) return;
    // Find IMAGE output slot of LoadImage and pixels input of VAEEncode.
    const outIdx = (loadNode.outputs || []).findIndex(o => /image/i.test(o?.name || "") || o?.type === "IMAGE");
    const inIdx  = (vaeEncode.inputs  || []).findIndex(i => /pixels/i.test(i?.name || "") || i?.type === "IMAGE");
    if (outIdx < 0 || inIdx < 0) return;
    try {
        loadNode.connect(outIdx, vaeEncode, inIdx);
        console.log("[MEC.DragRemix] Auto-wired LoadImage → VAEEncode");
    } catch (e) {
        console.warn("[MEC.DragRemix] auto-wire failed:", e);
    }
}

function _attachDnd() {
    const root = app.canvasEl || app.canvas?.canvas;
    if (!root || root._mecDragRemixBound) return;
    root._mecDragRemixBound = true;

    root.addEventListener("dragover", (e) => {
        if (e.dataTransfer?.types?.includes?.("Files")) {
            e.preventDefault();
            try { e.dataTransfer.dropEffect = "copy"; } catch { /* ignore */ }
        }
    });
    root.addEventListener("drop", async (e) => {
        const enabled = (() => {
            try { return app.ui.settings.getSettingValue("mec.drag_remix.enabled", true); }
            catch { return true; }
        })();
        if (!enabled) return;
        const files = e.dataTransfer?.files;
        if (!files || files.length === 0) return;
        const img = Array.from(files).find(f => /^image\//.test(f.type));
        if (!img) return;
        e.preventDefault();
        e.stopPropagation();
        const pos = _canvasPosFromEvent(e);
        try {
            const result = await _uploadImage(img);
            const fn = result.subfolder ? `${result.subfolder}/${result.name}` : result.name;
            const node = _spawnLoadImage(fn, pos);
            _autoWire(node);
        } catch (err) {
            console.warn("[MEC.DragRemix] drop failed:", err);
            alert(`Drag-Remix upload failed: ${err.message || err}`);
        }
    }, true);
}

app.registerExtension({
    name: "MEC.DragRemix",
    settings: [
        {
            id: "mec.drag_remix.enabled",
            name: "Drag-to-Remix: drop images onto canvas",
            tooltip: "Dropping image files spawns a LoadImage node at the drop position.",
            type: "boolean",
            default: true,
        },
        {
            id: "mec.drag_remix.auto_wire",
            name: "Drag-to-Remix: auto-wire into existing VAEEncode if present",
            type: "boolean",
            default: false,
        },
    ],
    async setup() {
        // ComfyUI already handles drag-drop of its own JSON workflows; we only act on raw images.
        // Use a short delay to ensure canvas element exists.
        setTimeout(_attachDnd, 500);
        console.log("[MEC.DragRemix] Loaded.");
    },
});
