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

async function _readJsonOrThrow(r) {
    const j = await r.json().catch(() => ({}));
    if (r.status === 409 && j.superseded) {
        const err = new Error("superseded");
        err.superseded = true;
        throw err;
    }
    if (!r.ok) {
        const err = new Error(j.error || `request failed (${r.status})`);
        err.status = r.status;
        err.needImage = j.need_image === true;
        err.needWeights = j.need_weights === true;
        throw err;
    }
    return j;
}

export async function samPredict(meta, imageBlobOrNull = null) {
    const doPost = async (body, headers) => {
        const r = await fetch(`${BASE}/sam`, { method: "POST", body, headers });
        if (r.status === 409) {
            const j = await r.json().catch(() => ({}));
            if (j.superseded) {
                const err = new Error("superseded");
                err.superseded = true;
                throw err;
            }
            if (j.need_image) {
                const err = new Error("need_image");
                err.needImage = true;
                throw err;
            }
            throw new Error(j.error || `SAM failed (${r.status})`);
        }
        if (!r.ok) {
            const j = await r.json().catch(() => ({}));
            throw new Error(j.error || `SAM failed (${r.status})`);
        }
        const score = parseFloat(r.headers.get("X-IME-Score") || "0");
        const blob = await r.blob();
        const bmp = await createImageBitmap(blob);
        const c = document.createElement("canvas");
        c.width = bmp.width;
        c.height = bmp.height;
        const cx = c.getContext("2d");
        cx.drawImage(bmp, 0, 0);
        bmp.close();
        const id = cx.getImageData(0, 0, c.width, c.height);
        const mask = new Uint8Array(c.width * c.height);
        for (let p = 0, i = 0; p < mask.length; p++, i += 4) mask[p] = id.data[i];
        return { mask, score, width: c.width, height: c.height };
    };

    if (imageBlobOrNull) {
        const fd = new FormData();
        fd.append("meta", JSON.stringify(meta));
        fd.append("image", imageBlobOrNull, "frame.png");
        return doPost(fd);
    }
    return doPost(JSON.stringify(meta), { "Content-Type": "application/json" });
}

export async function refine(meta, imageBlob, maskBlob, bandBlob) {
    const fd = new FormData();
    fd.append("meta", JSON.stringify(meta));
    fd.append("image", imageBlob, "image.png");
    fd.append("mask", maskBlob, "mask.png");
    fd.append("band", bandBlob, "band.png");
    const r = await fetch(`${BASE}/refine`, { method: "POST", body: fd });
    if (!r.ok) {
        const j = await r.json().catch(() => ({}));   // a body can be read once: one read for every error
        if (r.status === 409 && j.superseded) {
            const err = new Error("superseded");
            err.superseded = true;
            throw err;
        }
        const err = new Error(j.error || `refine failed (${r.status})`);
        err.status = r.status;
        err.needWeights = j.need_weights === true;
        throw err;
    }
    return r.blob();
}

export function release(editorId) {
    fetch(`${BASE}/release`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ editor_id: editorId }),
    }).catch(() => { /* fire-and-forget */ });
}

export async function samModels() {
    const r = await fetch(`${BASE}/sam/models`);
    return _readJsonOrThrow(r);
}
