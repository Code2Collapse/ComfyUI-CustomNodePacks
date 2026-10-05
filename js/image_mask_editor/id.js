/** editor_id generation, validation, duplicate detection on paste. */

import { copyStore } from "./api.js";

const ID_RE = /^[a-z0-9][a-z0-9-]{7,63}$/;

export function makeEditorId() {
    const raw = (crypto?.randomUUID?.() || _fallbackUuid()).toLowerCase();
    return raw.replace(/[^a-z0-9-]/g, "").slice(0, 36);
}

function _fallbackUuid() {
    return ([1e7] + -1e3 + -4e3 + -8e3 + -1e11).replace(/[018]/g, (c) =>
        (c ^ (crypto.getRandomValues(new Uint8Array(1))[0] & 15) >> (c / 4)).toString(16));
}

export function isValidEditorId(id) {
    return typeof id === "string" && ID_RE.test(id);
}

export function getEditorIdWidget(node) {
    return node.widgets?.find((w) => w.name === "editor_id");
}

export function ensureEditorId(node) {
    const w = getEditorIdWidget(node);
    if (!w) return "";
    if (!isValidEditorId(w.value)) {
        w.value = makeEditorId();
    }
    return w.value;
}

export function collectEditorIds(graph, skipNodeId = null) {
    const ids = new Set();
    const nodes = graph?._nodes || graph?.nodes || [];
    for (const n of nodes) {
        if (skipNodeId != null && n.id === skipNodeId) continue;
        const w = getEditorIdWidget(n);
        if (w?.value && isValidEditorId(w.value)) ids.add(w.value);
    }
    return ids;
}

export async function dedupeOnConfigure(node) {
    const w = getEditorIdWidget(node);
    if (!w?.value || !isValidEditorId(w.value)) {
        w.value = makeEditorId();
        return;
    }
    const graph = node.graph;
    if (!graph) return;
    const others = collectEditorIds(graph, node.id);
    if (!others.has(w.value)) return;
    const old = w.value;
    const neu = makeEditorId();
    w.value = neu;
    try {
        await copyStore(old, neu);
    } catch (e) {
        console.warn("[IME] copy on dedupe failed:", e);
    }
}
