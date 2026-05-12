/**
 * Shared upstream-frame discovery for video-aware mask nodes.
 *
 * Strategy (in priority order):
 *   1. Walk UPSTREAM from the node's "image" input. If any ancestor
 *      already has imgs[] (multi-frame batch preview), use those.
 *   2. If an ancestor exposes a videoEl/HTMLVideoElement (VHS_LoadVideo
 *      style), sample its frames via canvas.drawImage at uniform timestamps.
 *   3. Otherwise scan the ENTIRE graph for any node whose ancestry
 *      intersects the VME's ancestry (sibling preview) and pick the one
 *      with the most imgs[] — this catches the common
 *      "VHS_LoadVideo → [PreviewImage, VME]" pattern after a Queue Prompt.
 *   4. As a last resort, peek at the upstream node's widget "video"/"image"
 *      value and return a single-frame URL.
 */

import { app } from "../../scripts/app.js";

function _ancestorsOf(startId, maxDepth = 32) {
    const out = new Set();
    if (startId == null) return out;
    const queue = [{ id: startId, depth: 0 }];
    while (queue.length) {
        const { id, depth } = queue.shift();
        if (id == null || out.has(id)) continue;
        out.add(id);
        if (depth >= maxDepth) continue;
        const n = app.graph.getNodeById(id);
        if (!n?.inputs) continue;
        for (const inp of n.inputs) {
            if (inp.link == null) continue;
            const li = app.graph.links[inp.link];
            if (li) queue.push({ id: li.origin_id, depth: depth + 1 });
        }
    }
    return out;
}

async function _sampleVideoFrames(videoEl, n) {
    // Returns array of data: URLs of frames sampled uniformly across duration.
    const dur = isFinite(videoEl.duration) ? videoEl.duration : 0;
    if (dur <= 0) return [];
    const W = videoEl.videoWidth, H = videoEl.videoHeight;
    if (!W || !H) return [];
    const tmp = document.createElement("canvas");
    tmp.width = W; tmp.height = H;
    const tctx = tmp.getContext("2d");
    const out = [];
    for (let i = 0; i < n; i++) {
        const t = (n === 1) ? 0 : (i / (n - 1)) * dur;
        await new Promise((res) => {
            const onSeek = () => { videoEl.removeEventListener("seeked", onSeek); res(); };
            videoEl.addEventListener("seeked", onSeek);
            try { videoEl.currentTime = t; } catch { res(); }
            setTimeout(() => { videoEl.removeEventListener("seeked", onSeek); res(); }, 2000);
        });
        tctx.drawImage(videoEl, 0, 0, W, H);
        out.push(tmp.toDataURL("image/png"));
    }
    return out;
}

/**
 * Returns { urls: string[], dataUrls?: boolean } for the IMAGE feeding the node.
 * Async because video sampling is async. Existing callers can `await`.
 */
export async function findUpstreamFramesAsync(node, opts = {}) {
    const maxVideoFrames = opts.maxVideoFrames ?? 32;
    if (!node.inputs) return [];
    const inp = node.inputs.find(i => i.name === "image" && i.link != null);
    if (!inp) return [];
    const directLink = app.graph.links[inp.link];
    const sourceId = directLink?.origin_id;
    if (sourceId == null) return [];

    // Pass 1: walk UPSTREAM directly.
    {
        const visited = new Set();
        const queue = [{ id: sourceId, depth: 0 }];
        while (queue.length) {
            const { id, depth } = queue.shift();
            if (id == null || visited.has(id)) continue;
            visited.add(id);
            const n = app.graph.getNodeById(id);
            if (!n) continue;

            // Multi-frame preview thumbnails (batch).
            if (n.imgs?.length > 1) return n.imgs.map(im => im.src);

            // Single-frame thumbnails — keep as fallback but keep scanning
            // for something with more frames.
            if (n.imgs?.length === 1 && depth === 0) {
                // remember but don't return yet — a sibling preview may have more
            }

            // VHS_LoadVideo-style video element.
            const vid = n.videoEl || n.videoContainer?.querySelector?.("video") || null;
            if (vid && vid.readyState >= 2) {
                try {
                    const frames = await _sampleVideoFrames(vid, maxVideoFrames);
                    if (frames.length) return frames;
                } catch (e) { /* fall through */ }
            }

            if (depth < 6 && n.inputs) {
                for (const i2 of n.inputs) {
                    if (i2.link == null) continue;
                    const li = app.graph.links[i2.link];
                    if (li) queue.push({ id: li.origin_id, depth: depth + 1 });
                }
            }
        }
    }

    // Pass 2: sibling-preview scan. Build VME's ancestor set,
    // then pick the graph node whose ancestry intersects ours
    // and has the most imgs[].
    const myAncestors = _ancestorsOf(sourceId);
    myAncestors.add(sourceId);
    let best = null;
    for (const n of app.graph._nodes) {
        if (!n.imgs?.length) continue;
        if (n.id === node.id) continue;
        // Find this node's own ancestors (going upstream from its inputs).
        let nAnc = null;
        for (const i2 of n.inputs || []) {
            if (i2.link == null) continue;
            const li = app.graph.links[i2.link];
            if (li) {
                nAnc = nAnc || new Set();
                for (const a of _ancestorsOf(li.origin_id)) nAnc.add(a);
            }
        }
        if (!nAnc) continue;
        // Intersection test.
        let shares = false;
        for (const a of myAncestors) {
            if (nAnc.has(a)) { shares = true; break; }
        }
        if (!shares) continue;
        if (!best || n.imgs.length > best.imgs.length) best = n;
    }
    if (best?.imgs?.length) return best.imgs.map(im => im.src);

    // Pass 3: upstream single-frame fallback (imgs[0] or widget value).
    {
        const visited = new Set();
        const queue = [{ id: sourceId, depth: 0 }];
        while (queue.length) {
            const { id, depth } = queue.shift();
            if (id == null || visited.has(id)) continue;
            visited.add(id);
            const n = app.graph.getNodeById(id);
            if (!n) continue;
            if (n.imgs?.length === 1) return n.imgs.map(im => im.src);
            const w = n.widgets?.find(w => w.name === "image" || w.name === "video");
            if (w?.value && typeof w.value === "string") {
                const parts = w.value.split("/");
                const sub = parts.length > 1 ? parts.slice(0, -1).join("/") : "";
                const fn = parts[parts.length - 1];
                return [`/view?filename=${encodeURIComponent(fn)}&subfolder=${encodeURIComponent(sub)}&type=input`];
            }
            if (depth < 6 && n.inputs) {
                for (const i2 of n.inputs) {
                    if (i2.link == null) continue;
                    const li = app.graph.links[i2.link];
                    if (li) queue.push({ id: li.origin_id, depth: depth + 1 });
                }
            }
        }
    }

    return [];
}

/** Synchronous wrapper that returns [] until the async resolution completes.
 *  Prefer findUpstreamFramesAsync in new code. */
export function findUpstreamFrames(node) {
    // Best-effort sync path: just pass 1+2+3 without video sampling.
    if (!node.inputs) return [];
    const inp = node.inputs.find(i => i.name === "image" && i.link != null);
    if (!inp) return [];
    const directLink = app.graph.links[inp.link];
    const sourceId = directLink?.origin_id;
    if (sourceId == null) return [];

    // Pass 1: upstream multi-frame
    {
        const visited = new Set();
        const queue = [{ id: sourceId, depth: 0 }];
        while (queue.length) {
            const { id, depth } = queue.shift();
            if (id == null || visited.has(id)) continue;
            visited.add(id);
            const n = app.graph.getNodeById(id);
            if (!n) continue;
            if (n.imgs?.length > 1) return n.imgs.map(im => im.src);
            if (depth < 6 && n.inputs) {
                for (const i2 of n.inputs) {
                    if (i2.link == null) continue;
                    const li = app.graph.links[i2.link];
                    if (li) queue.push({ id: li.origin_id, depth: depth + 1 });
                }
            }
        }
    }

    // Pass 2: sibling preview scan
    const myAncestors = _ancestorsOf(sourceId);
    myAncestors.add(sourceId);
    let best = null;
    for (const n of app.graph._nodes) {
        if (!n.imgs?.length) continue;
        if (n.id === node.id) continue;
        let nAnc = null;
        for (const i2 of n.inputs || []) {
            if (i2.link == null) continue;
            const li = app.graph.links[i2.link];
            if (li) {
                nAnc = nAnc || new Set();
                for (const a of _ancestorsOf(li.origin_id)) nAnc.add(a);
            }
        }
        if (!nAnc) continue;
        let shares = false;
        for (const a of myAncestors) {
            if (nAnc.has(a)) { shares = true; break; }
        }
        if (!shares) continue;
        if (!best || n.imgs.length > best.imgs.length) best = n;
    }
    if (best?.imgs?.length) return best.imgs.map(im => im.src);

    // Pass 3: single-frame fallback
    {
        const visited = new Set();
        const queue = [{ id: sourceId, depth: 0 }];
        while (queue.length) {
            const { id, depth } = queue.shift();
            if (id == null || visited.has(id)) continue;
            visited.add(id);
            const n = app.graph.getNodeById(id);
            if (!n) continue;
            if (n.imgs?.length === 1) return n.imgs.map(im => im.src);
            const w = n.widgets?.find(w => w.name === "image" || w.name === "video");
            if (w?.value && typeof w.value === "string") {
                const parts = w.value.split("/");
                const sub = parts.length > 1 ? parts.slice(0, -1).join("/") : "";
                const fn = parts[parts.length - 1];
                return [`/view?filename=${encodeURIComponent(fn)}&subfolder=${encodeURIComponent(sub)}&type=input`];
            }
            if (depth < 6 && n.inputs) {
                for (const i2 of n.inputs) {
                    if (i2.link == null) continue;
                    const li = app.graph.links[i2.link];
                    if (li) queue.push({ id: li.origin_id, depth: depth + 1 });
                }
            }
        }
    }
    return [];
}
