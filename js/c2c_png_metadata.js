// c2c_png_metadata.js — safe PNG text-chunk reader + workflow parse (shared)
import { app } from "../../scripts/app.js";

const PNG_SIG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

function _utf8(arr) {
    try { return new TextDecoder("utf-8").decode(arr); } catch { return ""; }
}

function _latin1(arr) {
    try { return new TextDecoder("latin1").decode(arr); } catch { return ""; }
}

/** Bounds-safe PNG tEXt/iTXt reader; skips malformed chunks instead of throwing. */
export function readPngTextChunks(buffer) {
    const out = {};
    if (!buffer || buffer.byteLength < 8) return out;
    const view = new DataView(buffer);
    for (let i = 0; i < 8; i++) {
        if (view.getUint8(i) !== PNG_SIG[i]) return out;
    }
    let off = 8;
    const end = view.byteLength;
    while (off + 12 <= end) {
        const length = view.getUint32(off);
        if (length > end || off + 12 + length > end) break;
        const type = String.fromCharCode(
            view.getUint8(off + 4), view.getUint8(off + 5),
            view.getUint8(off + 6), view.getUint8(off + 7),
        );
        const data = new Uint8Array(buffer, off + 8, length);
        off += 12 + length;
        try {
            if (type === "tEXt") {
                const nul = data.indexOf(0);
                if (nul > 0) {
                    const key = _latin1(data.subarray(0, nul));
                    const val = _latin1(data.subarray(nul + 1));
                    if (key) out[key] = val;
                }
            } else if (type === "iTXt") {
                const nul = data.indexOf(0);
                if (nul > 0) {
                    const key = _utf8(data.subarray(0, nul));
                    let p = nul + 1;
                    if (p + 2 > data.length) continue;
                    const compFlag = data[p]; p += 2;
                    while (p < data.length && data[p] !== 0) p++;
                    p++;
                    while (p < data.length && data[p] !== 0) p++;
                    p++;
                    const text = compFlag === 0 ? _utf8(data.subarray(p)) : "";
                    if (key) out[key] = text || (compFlag ? "<compressed>" : "");
                }
            }
        } catch (_) { /* skip bad chunk */ }
        if (type === "IEND") break;
    }
    return out;
}

export async function readImageMetadata(file) {
    if (!file) return {};
    const name = (file.name || "").toLowerCase();
    if (name.endsWith(".png") || file.type === "image/png") {
        try {
            return readPngTextChunks(await file.arrayBuffer());
        } catch { return {}; }
    }
    return {};
}

const WORKFLOW_KEYS = ["workflow", "Workflow", "prompt", "Prompt"];

/**
 * @returns {{ status: "none"|"valid"|"corrupt", kind?: string, key?: string, data?: any, error?: string }}
 */
export function assessWorkflowMetadata(meta) {
    if (!meta || !Object.keys(meta).length) return { status: "none" };
    for (const k of WORKFLOW_KEYS) {
        const raw = meta[k];
        if (!raw || typeof raw !== "string" || !raw.trim()) continue;
        try {
            const data = JSON.parse(raw);
            const kind = (k.toLowerCase() === "prompt") ? "comfy-prompt" : "comfy-workflow";
            return { status: "valid", kind, key: k, data };
        } catch (e) {
            return { status: "corrupt", key: k, error: String(e?.message || e) };
        }
    }
    if (meta.parameters && String(meta.parameters).trim()) {
        return { status: "valid", kind: "a1111", key: "parameters", data: meta.parameters };
    }
    return { status: "none" };
}

export function toast(message, type = "info") {
    try {
        app.ui?.showToast?.({ message, type, duration: 4000 });
    } catch (_) {
        console.info("[C2C]", message);
    }
}

/** Upload dropped file and spawn LoadImage at drop coords. Does not touch workflow JSON. */
export async function loadImageOnly(file, dropEvent) {
    const { api } = await import("../../scripts/api.js");
    const body = new FormData();
    body.append("image", file);
    body.append("overwrite", "true");
    const resp = await api.fetchApi("/upload/image", { method: "POST", body });
    if (!resp.ok) {
        toast(`Could not upload ${file.name}`, "error");
        return false;
    }
    const uploaded = await resp.json();
    const sub = uploaded.subfolder ? `${uploaded.subfolder}/` : "";
    const imageName = `${sub}${uploaded.name}`;

    const node = LiteGraph.createNode("LoadImage");
    if (!node) {
        toast("LoadImage node type not registered", "error");
        return false;
    }

    let pos = [80, 80];
    const evt = dropEvent || app.__c2cLastDropEvent;
    const canvas = app.canvas;
    if (evt && canvas?.canvas) {
        const rect = canvas.canvas.getBoundingClientRect();
        const ds = canvas.ds;
        pos = [
            (evt.clientX - rect.left - ds.offset[0]) / ds.scale,
            (evt.clientY - rect.top - ds.offset[1]) / ds.scale,
        ];
    }
    node.pos = pos;

    const imgW = node.widgets?.find((w) => w.name === "image");
    if (imgW) {
        imgW.value = imageName;
        try { imgW.callback?.(imageName, node, imgW); } catch (_) {}
    }
    app.graph.add(node);
    canvas?.selectNode?.(node);
    canvas?.setDirty?.(true, true);
    toast(`Loaded image only: ${file.name}`, "info");
    return true;
}
