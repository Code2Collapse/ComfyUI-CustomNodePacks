/** HTTP client for /c2c/image_mask_editor/* routes. */

const BASE = "/c2c/image_mask_editor";

export async function fetchState(editorId) {
    const r = await fetch(`${BASE}/state?id=${encodeURIComponent(editorId)}`);
    if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        throw new Error(j.error || `state failed (${r.status})`);
    }
    return r.json();
}

export async function fetchFramePng(editorId, frame) {
    const r = await fetch(
        `${BASE}/frame?id=${encodeURIComponent(editorId)}&frame=${frame | 0}`,
    );
    if (r.status === 404) return null;
    if (!r.ok) throw new Error(`frame GET failed (${r.status})`);
    return r.blob();
}

export async function putFramePng(editorId, frame, pngBlob) {
    const r = await fetch(
        `${BASE}/frame?id=${encodeURIComponent(editorId)}&frame=${frame | 0}`,
        { method: "POST", body: pngBlob },
    );
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || `frame POST failed (${r.status})`);
    return j;
}

export async function deleteFrame(editorId, frame) {
    const r = await fetch(
        `${BASE}/frame?id=${encodeURIComponent(editorId)}&frame=${frame | 0}`,
        { method: "DELETE" },
    );
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || `frame DELETE failed (${r.status})`);
    return j;
}

export async function clearStore(editorId) {
    const r = await fetch(`${BASE}/clear?id=${encodeURIComponent(editorId)}`, { method: "POST" });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || `clear failed (${r.status})`);
    return j;
}

export async function copyStore(fromId, toId) {
    const r = await fetch(
        `${BASE}/copy?from=${encodeURIComponent(fromId)}&to=${encodeURIComponent(toId)}`,
        { method: "POST" },
    );
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || `copy failed (${r.status})`);
    return j;
}
