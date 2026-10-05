/**
 * Shared upstream-frame discovery for video-aware mask nodes.
 *
 * Strategy (in priority order):
 *   1. Direct loader on the `image` input → POST /c2c/frames/plan (exact batch).
 *   2. Walk UPSTREAM: multi-frame imgs[] preview batch.
 *   3. Sibling preview scan (VHS_LoadVideo → [PreviewImage, VME]).
 *   4. Video element uniform sampling (capped by c2c.vme.sampledFrames).
 *   5. Single-frame fallback (imgs[0] or widget value).
 */

import { app } from "../../scripts/app.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";

export const PLAN_LOADER_TYPES = new Set([
    "LoadVideoC2C",
    "LoadVideoPathC2C",
    "LoadImagesPathC2C",
    "VHS_LoadVideo",
    "VHS_LoadVideoPath",
    "VHS_LoadImagesPath",
]);

const SAMPLE_NOTE =
    "Showing N evenly spaced frames from the video preview — the batch this node receives may differ. " +
    "Connect a C2C or VHS loader directly, or queue once, for exact frames.";

function _settingInt(id, fallback, min, max) {
    try {
        const v = Number(app.ui?.settings?.getSettingValue?.(id, fallback));
        if (!Number.isFinite(v)) return fallback;
        return Math.max(min, Math.min(max, Math.round(v)));
    } catch {
        return fallback;
    }
}

function _nodeType(node) {
    return node?.comfyClass || node?.type || "";
}

function _widgetValue(node, name) {
    const w = node.widgets?.find((x) => x.name === name);
    return w != null ? w.value : undefined;
}

function _collectLoaderWidgets(node, nodeType) {
    const widgets = {};
    if (nodeType === "LoadImagesPathC2C" || nodeType === "VHS_LoadImagesPath") {
        widgets.directory = _widgetValue(node, "directory");
        widgets.image_load_cap = _widgetValue(node, "image_load_cap") ?? 0;
        widgets.skip_first_images = _widgetValue(node, "skip_first_images") ?? 0;
        widgets.select_every_nth = _widgetValue(node, "select_every_nth") ?? 1;
        return widgets;
    }
    widgets.video = _widgetValue(node, "video");
    widgets.force_rate = _widgetValue(node, "force_rate") ?? 0;
    widgets.skip_first_frames = _widgetValue(node, "skip_first_frames") ?? 0;
    widgets.select_every_nth = _widgetValue(node, "select_every_nth") ?? 1;
    widgets.frame_load_cap = _widgetValue(node, "frame_load_cap") ?? 0;
    widgets.format = _widgetValue(node, "format") ?? "None";
    widgets.custom_width = _widgetValue(node, "custom_width") ?? 0;
    widgets.custom_height = _widgetValue(node, "custom_height") ?? 0;
    if (nodeType === "LoadVideoPathC2C" || nodeType === "VHS_LoadVideoPath") {
        const sf = _widgetValue(node, "sequence_fps");
        if (sf != null) widgets.sequence_fps = sf;
    }
    return widgets;
}

// Executed preview images of a node. Frontend 1.52.7 keeps them in app.nodeOutputs[id].images and
// fills node.imgs only when the node is DRAWN, so a preview node that is off screen (or not yet
// painted) has no imgs while its images exist (measured 2026-10-05: 32 in nodeOutputs, imgs null).
function _executedUrls(n) {
    if (n?.imgs?.length) return n.imgs.map((im) => im.src);
    const out = app.nodeOutputs?.[String(n?.id)] ?? app.nodeOutputs?.[n?.id];
    const list = Array.isArray(out?.images) ? out.images : [];
    return list.filter((im) => im && im.filename).map((im) =>
        `/view?filename=${encodeURIComponent(im.filename)}&subfolder=${encodeURIComponent(im.subfolder || "")}` +
        `&type=${encodeURIComponent(im.type || "output")}`);
}

function _withDims(src, n) {
    const im0 = n?.imgs?.[0];
    if (im0?.naturalWidth) {        // else the editor learns them from the first loaded frame
        src.width = im0.naturalWidth;
        src.height = im0.naturalHeight;
    }
    return src;
}

function _ancestorsOf(graph, startId, maxDepth = 32) {
    const out = new Set();
    if (startId == null || !graph) return out;
    const queue = [{ id: startId, depth: 0 }];
    while (queue.length) {
        const { id, depth } = queue.shift();
        if (id == null || out.has(id)) continue;
        out.add(id);
        if (depth >= maxDepth) continue;
        const n = graph.getNodeById?.(id);
        if (!n?.inputs) continue;
        for (const inp of n.inputs) {
            if (inp.link == null) continue;
            const li = graph.links?.[inp.link];
            if (li) queue.push({ id: li.origin_id, depth: depth + 1 });
        }
    }
    return out;
}

async function _sampleVideoFrames(videoEl, n) {
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

function _sourceFromUrls(urls, kind, sampled, note) {
    const count = urls.length;
    return {
        kind,
        count,
        width: 0,
        height: 0,
        url(i) { return urls[i]; },
        sampled: !!sampled,
        note: note || "",
    };
}

async function _planFromDirectLoader(node, nodeType) {
    const widgets = _collectLoaderWidgets(node, nodeType);
    const resp = await fetch("/c2c/frames/plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node_type: nodeType, widgets }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
        throw new Error(data.error || `Frame plan failed (${resp.status}).`);
    }
    const token = data.token;
    const count = data.count | 0;
    const width = data.width | 0;
    const height = data.height | 0;
    return {
        kind: "plan",
        count,
        width,
        height,
        url(i) {
            return `/c2c/frames/thumb?token=${encodeURIComponent(token)}&i=${i | 0}&max=384&fmt=jpeg`;
        },
        sampled: false,
        note: "",
    };
}

/**
 * Discover the frame source feeding `node`'s `image` input.
 * @returns {Promise<{kind:string,count:number,width:number,height:number,url:Function,sampled:boolean,note:string}>}
 */
export async function findUpstreamFrameSource(node, opts = {}) {
    if (!node.inputs) {
        return { kind: "none", count: 0, width: 0, height: 0, url: () => "", sampled: false, note: "" };
    }
    const inp = node.inputs.find((i) => i.name === "image" && i.link != null);
    if (!inp) {
        return { kind: "none", count: 0, width: 0, height: 0, url: () => "", sampled: false, note: "" };
    }
    const graph = node.graph || app.graph;
    const directLink = graph.links?.[inp.link];
    const sourceId = directLink?.origin_id;
    if (sourceId == null) {
        return { kind: "none", count: 0, width: 0, height: 0, url: () => "", sampled: false, note: "" };
    }

    const directNode = graph.getNodeById?.(sourceId);
    const directType = _nodeType(directNode);
    if (directNode && PLAN_LOADER_TYPES.has(directType)) {
        try {
            return await _planFromDirectLoader(directNode, directType);
        } catch (e) {
            __c2cReport("_frame_finder.plan", e);
        }
    }

    const maxVideoFrames = opts.maxVideoFrames ??
        _settingInt("c2c.vme.sampledFrames", 32, 2, 2000);

    // Pass 1: walk UPSTREAM directly.
    {
        const visited = new Set();
        const queue = [{ id: sourceId, depth: 0 }];
        while (queue.length) {
            const { id, depth } = queue.shift();
            if (id == null || visited.has(id)) continue;
            visited.add(id);
            const n = graph.getNodeById?.(id);
            if (!n) continue;

            const executed = _executedUrls(n);
            if (executed.length > 1) {
                return _withDims(_sourceFromUrls(executed, "previews", false, ""), n);
            }

            const vid = n.videoEl || n.videoContainer?.querySelector?.("video") || null;
            if (vid && vid.readyState >= 2) {
                try {
                    const frames = await _sampleVideoFrames(vid, maxVideoFrames);
                    if (frames.length) {
                        const note = SAMPLE_NOTE.replace("N", String(frames.length));
                        const src = _sourceFromUrls(frames, "sampled", true, note);
                        src.width = vid.videoWidth || 0;
                        src.height = vid.videoHeight || 0;
                        return src;
                    }
                } catch (e) { __c2cReport("_frame_finder", e); }
            }

            if (depth < 6 && n.inputs) {
                for (const i2 of n.inputs) {
                    if (i2.link == null) continue;
                    const li = graph.links?.[i2.link];
                    if (li) queue.push({ id: li.origin_id, depth: depth + 1 });
                }
            }
        }
    }

    // Pass 2: sibling-preview scan (VHS_LoadVideo -> [PreviewImage, VME]). A preview qualifies only
    // when it shows an image this node's input derives from: one of its inputs comes STRAIGHT from the
    // source or an ancestor of it, and none passes through this node. Sharing any ancestor is not
    // enough - a preview DOWNSTREAM of this node shows its output (the mask), and an editor that took
    // it as the plate drew on its own last result (measured 2026-10-05, L7.38: the second session of
    // LoadImage -> ImageMaskEditor -> MaskToImage -> PreviewImage edited the white-on-black mask).
    const myAncestors = _ancestorsOf(graph, sourceId);
    myAncestors.add(sourceId);
    let best = null;
    let bestUrls = [];
    const allNodes = graph._nodes || graph.nodes || [];
    for (const n of allNodes) {
        if (n.id === node.id) continue;
        const urls = _executedUrls(n);
        if (!urls.length) continue;
        let fedFromMyChain = false;
        let downstreamOfMe = false;
        for (const i2 of n.inputs || []) {
            if (i2.link == null) continue;
            const li = graph.links?.[i2.link];
            if (!li) continue;
            if (myAncestors.has(li.origin_id)) fedFromMyChain = true;
            if (li.origin_id === node.id || _ancestorsOf(graph, li.origin_id).has(node.id)) downstreamOfMe = true;
        }
        if (!fedFromMyChain || downstreamOfMe) continue;
        if (!best || urls.length > bestUrls.length) { best = n; bestUrls = urls; }
    }
    if (bestUrls.length) {
        return _withDims(_sourceFromUrls(bestUrls, "previews", false, ""), best);
    }

    // Pass 3: single-frame fallback.
    {
        const visited = new Set();
        const queue = [{ id: sourceId, depth: 0 }];
        while (queue.length) {
            const { id, depth } = queue.shift();
            if (id == null || visited.has(id)) continue;
            visited.add(id);
            const n = graph.getNodeById?.(id);
            if (!n) continue;
            const executed = _executedUrls(n);
            if (executed.length === 1) {
                return _withDims(_sourceFromUrls(executed, "single", false, ""), n);
            }
            const w = n.widgets?.find((w) => w.name === "image" || w.name === "video");
            if (w?.value && typeof w.value === "string") {
                const parts = w.value.split("/");
                const sub = parts.length > 1 ? parts.slice(0, -1).join("/") : "";
                const fn = parts[parts.length - 1];
                const url = `/view?filename=${encodeURIComponent(fn)}&subfolder=${encodeURIComponent(sub)}&type=input`;
                return _sourceFromUrls([url], "single", false, "");
            }
            if (depth < 6 && n.inputs) {
                for (const i2 of n.inputs) {
                    if (i2.link == null) continue;
                    const li = graph.links?.[i2.link];
                    if (li) queue.push({ id: li.origin_id, depth: depth + 1 });
                }
            }
        }
    }

    return { kind: "none", count: 0, width: 0, height: 0, url: () => "", sampled: false, note: "" };
}

/**
 * Returns frame URL strings for the IMAGE feeding the node.
 * Async because video sampling / frame planning is async.
 */
export async function findUpstreamFramesAsync(node, opts = {}) {
    const src = await findUpstreamFrameSource(node, opts);
    if (!src.count) return [];
    return Array.from({ length: src.count }, (_, i) => src.url(i));
}

/** Synchronous wrapper that returns [] until the async resolution completes.
 *  Prefer findUpstreamFramesAsync in new code. */
export function findUpstreamFrames(node) {
    if (!node.inputs) return [];
    const inp = node.inputs.find(i => i.name === "image" && i.link != null);
    if (!inp) return [];
    const graph = node.graph || app.graph;
    const directLink = graph.links?.[inp.link];
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
            const n = graph.getNodeById?.(id);
            if (!n) continue;
            const executed = _executedUrls(n);
            if (executed.length > 1) return executed;
            if (depth < 6 && n.inputs) {
                for (const i2 of n.inputs) {
                    if (i2.link == null) continue;
                    const li = graph.links?.[i2.link];
                    if (li) queue.push({ id: li.origin_id, depth: depth + 1 });
                }
            }
        }
    }

    // Pass 2: sibling preview scan
    const myAncestors = _ancestorsOf(graph, sourceId);
    myAncestors.add(sourceId);
    let bestUrls = [];
    const allNodes = graph._nodes || graph.nodes || [];
    for (const n of allNodes) {
        const urls = _executedUrls(n);
        if (!urls.length) continue;
        if (n.id === node.id) continue;
        let nAnc = null;
        for (const i2 of n.inputs || []) {
            if (i2.link == null) continue;
            const li = graph.links?.[i2.link];
            if (li) {
                nAnc = nAnc || new Set();
                for (const a of _ancestorsOf(graph, li.origin_id)) nAnc.add(a);
            }
        }
        if (!nAnc) continue;
        let shares = false;
        for (const a of myAncestors) {
            if (nAnc.has(a)) { shares = true; break; }
        }
        if (!shares) continue;
        if (urls.length > bestUrls.length) bestUrls = urls;
    }
    if (bestUrls.length) return bestUrls;

    // Pass 3: single-frame fallback
    {
        const visited = new Set();
        const queue = [{ id: sourceId, depth: 0 }];
        while (queue.length) {
            const { id, depth } = queue.shift();
            if (id == null || visited.has(id)) continue;
            visited.add(id);
            const n = graph.getNodeById?.(id);
            if (!n) continue;
            const executed = _executedUrls(n);
            if (executed.length === 1) return executed;
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
                    const li = graph.links?.[i2.link];
                    if (li) queue.push({ id: li.origin_id, depth: depth + 1 });
                }
            }
        }
    }
    return [];
}
