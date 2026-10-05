/** Resolve frame source for ImageMaskEditorC2C (loaders, last run, upstream). */
import { app } from "../../../scripts/app.js";
import { PLAN_LOADER_TYPES, findUpstreamFrameSource } from "../_frame_finder.js";

const RUN_NOTE = "Frames from the last run - queue again after changing the input.";
const UPSTREAM_NOTE =
    "Showing the nearest upstream preview - run once so the editor shows exactly what this node receives.";

function _directUpstream(node) {
    const inp = node.inputs?.find((i) => i.name === "image" && i.link != null);
    if (!inp) return null;
    const graph = node.graph || app.graph;
    const link = graph.links?.[inp.link];
    if (!link) return null;
    return graph.getNodeById?.(link.origin_id) ?? null;
}

function _nodeType(node) {
    return node?.comfyClass || node?.type || "";
}

function _runPayload(node) {
    if (node?._imeRun?.count > 0) return node._imeRun;
    const out = app.nodeOutputs?.[String(node.id)] ?? app.nodeOutputs?.[node.id];
    const p = out?.c2c_ime_frames?.[0];
    return p?.count > 0 ? p : null;
}

function _runSource(payload) {
    const sub = encodeURIComponent(payload.subfolder || "");
    const typ = encodeURIComponent(payload.type || "temp");
    const ver = encodeURIComponent(payload.version || "");
    const count = payload.count | 0;
    const pad = (i) => String(i | 0).padStart(5, "0");
    return {
        kind: "run",
        count,
        width: payload.width | 0,
        height: payload.height | 0,
        url(i) {
            return `/view?filename=${pad(i)}.png&subfolder=${sub}&type=${typ}&v=${ver}`;
        },
        thumbUrl(i) {
            return `/view?filename=t_${pad(i)}.jpg&subfolder=${sub}&type=${typ}&v=${ver}`;
        },
        sampled: false,
        note: RUN_NOTE,
    };
}

/**
 * @returns {Promise<{kind:string,count:number,width:number,height:number,url:Function,sampled:boolean,note:string,thumbUrl?:Function}>}
 */
export async function resolveEditorSource(node) {
    const direct = _directUpstream(node);
    const directType = _nodeType(direct);
    if (direct && (PLAN_LOADER_TYPES.has(directType) || directType === "LoadImage")) {
        return findUpstreamFrameSource(node);
    }

    const run = _runPayload(node);
    if (run) return _runSource(run);

    const src = await findUpstreamFrameSource(node);
    if (src.count > 0 && !src.note) src.note = UPSTREAM_NOTE;
    return src;
}
