/**
 * How the C2C viewer talks to the server (L7.114; owner answer round 5: hidden helpers through ComfyUI's own queue).
 *
 * No new HTTP route. Each interactive step builds a small HELPER PROMPT in the browser - the nodes that feed the
 * Matte node (cut out of app.graphToPrompt()), plus one hidden helper node - and posts it to /prompt at the front of
 * the queue. The answer is the helper's `ui` output ("c2c_viewer"), read from the websocket `executed` event; the
 * listeners are attached BEFORE the post (a fully cached helper finishes before the POST returns) and events are
 * buffered until the prompt id is known. /history is polled once a second as a backstop while - and only while - a
 * helper is pending, and /queue tells the user when the helper waits behind another job. Images come back as temp
 * files through /view.
 *
 * Pure functions (prompt building, ids) and the runner take the ComfyUI `api` object as a parameter, so this module
 * imports nothing from ComfyUI and runs under Node (tests/test_viewer_core_js.py).
 *
 * License: Apache-2.0
 */

export const HELPERS = {
    plate: "C2CViewerPlateProxy",
    detect: "C2CViewerDetect",
    result: "C2CViewerResult",
};
export const UI_KEY = "c2c_viewer";
const HELPER_ID = "c2c_viewer_helper";

/** A link in an API prompt: [source node id, output index]. */
export function isLink(v) {
    return Array.isArray(v) && v.length === 2 && (typeof v[0] === "string" || typeof v[0] === "number")
        && Number.isInteger(v[1]);
}

/** Every node id upstream of ``startIds`` (through any input), not counting the start ids themselves. */
export function ancestorsOf(prompt, startIds, onlyInputs = null) {
    const seen = new Set();
    const stack = [];
    for (const id of startIds) {
        const node = prompt[String(id)];
        if (!node) continue;
        for (const [name, v] of Object.entries(node.inputs || {})) {
            if (onlyInputs && !onlyInputs.includes(name)) continue;
            if (isLink(v)) stack.push(String(v[0]));
        }
    }
    while (stack.length) {
        const id = stack.pop();
        if (seen.has(id) || !prompt[id]) continue;
        seen.add(id);
        for (const v of Object.values(prompt[id].inputs || {})) if (isLink(v)) stack.push(String(v[0]));
    }
    return seen;
}

function pick(prompt, ids) {
    const out = {};
    for (const id of ids) out[id] = JSON.parse(JSON.stringify(prompt[id]));
    return out;
}

/** The Matte node's id in an API prompt. Inside a subgraph ComfyUI flattens ids to "wrapper:inner"; ``execId`` is
 *  the id the viewer computed (executionIdOf), ``localId`` the node's own id, used when the computed one is absent. */
export function findNodeId(prompt, classType, execId, localId) {
    if (execId != null && prompt[String(execId)]?.class_type === classType) return String(execId);
    const local = String(localId);
    const hits = Object.keys(prompt).filter((k) => prompt[k]?.class_type === classType
        && (k === local || k.endsWith(":" + local)));
    return hits.length === 1 ? hits[0] : null;
}

/**
 * The helper prompt for one viewer step.
 *  kind "plate":  the nodes feeding the Matte node's `image` + Plate Proxy
 *  kind "detect": the same + Detect (opts: frame, model, text, points, boxes)
 *  kind "render": everything feeding the Matte node + the Matte node itself with ``opts.picks`` + Viewer Result
 *                 (opts: picks, start)
 * Returns {prompt, helperId}. Throws an Error with a sentence when the graph cannot give a plate.
 */
export function helperPrompt(prompt, matteId, kind, opts = {}) {
    const matte = prompt[String(matteId)];
    if (!matte) throw new Error("The Mask + Matting node is not in the graph that would run (is it muted or bypassed?).");
    const image = matte.inputs?.image;
    if (!isLink(image)) throw new Error("Connect a plate to the node's image input first.");
    const helperId = HELPER_ID;
    if (kind === "plate" || kind === "detect") {
        const ids = ancestorsOf(prompt, [String(matteId)], ["image"]);
        const out = pick(prompt, ids);
        out[helperId] = kind === "plate"
            ? { class_type: HELPERS.plate, inputs: { image, max_side: opts.maxSide ?? 960, max_frames: opts.maxFrames ?? 240 } }
            : { class_type: HELPERS.detect, inputs: {
                image, frame: opts.frame | 0, model: String(opts.model || ""), text: String(opts.text || ""),
                points: opts.points ? JSON.stringify(opts.points) : "", boxes: opts.boxes ? JSON.stringify(opts.boxes) : "",
                max_side: opts.maxSide ?? 960 } };
        return { prompt: out, helperId };
    }
    if (kind === "render") {
        const ids = ancestorsOf(prompt, [String(matteId)]);
        ids.add(String(matteId));
        const out = pick(prompt, ids);
        out[String(matteId)].inputs.picks = String(opts.picks || "");
        out[helperId] = { class_type: HELPERS.result, inputs: {
            image, alpha: [String(matteId), 1], coarse: [String(matteId), 0], start: opts.start | 0,
            max_side: opts.maxSide ?? 960, max_frames: opts.maxFrames ?? 120 } };
        return { prompt: out, helperId };
    }
    throw new Error(`Unknown viewer step '${kind}'.`);
}

function nodesOf(graph) {
    if (!graph) return [];
    if (Array.isArray(graph._nodes)) return graph._nodes;
    if (Array.isArray(graph.nodes)) return graph.nodes;
    return [];
}

function innerGraph(n) {
    const g = n?.subgraph;
    return g && typeof g === "object" && (Array.isArray(g._nodes) || Array.isArray(g.nodes)) ? g : null;
}

/** The execution id ComfyUI gives ``node``: "12" at the top level, "7:12" (wrapper:inner, nested deeper with more
 *  colons) inside subgraphs. A subgraph used by several wrappers gives the first wrapper path found. */
export function executionIdOf(rootGraph, node) {
    if (!node) return null;
    if (!node.graph || node.graph === rootGraph) return String(node.id);
    const stack = [{ graph: rootGraph, path: [] }];
    const seen = new Set();
    while (stack.length) {
        const { graph, path } = stack.pop();
        if (seen.has(graph)) continue;
        seen.add(graph);
        for (const n of nodesOf(graph)) {
            const inner = innerGraph(n);
            if (!inner) continue;
            const p = [...path, n.id];
            if (inner === node.graph) return [...p, node.id].join(":");
            stack.push({ graph: inner, path: p });
        }
    }
    return null;
}

/** ComfyUI's validation answer (HTTP 400 from /prompt) as one sentence. */
export function validationMessage(body) {
    const err = body?.error;
    const parts = [];
    if (err?.message) parts.push(String(err.message) + (err.details ? `: ${err.details}` : ""));
    for (const [id, ne] of Object.entries(body?.node_errors || {})) {
        for (const e of ne?.errors || []) parts.push(`${ne.class_type || id}: ${e.message}${e.details ? ` (${e.details})` : ""}`);
    }
    return parts.length ? `ComfyUI refused the viewer's job. ${parts.join("; ")}` : "ComfyUI refused the viewer's job.";
}

/**
 * Run helper prompts. ``api`` needs addEventListener / removeEventListener (CustomEvent with .detail), fetchApi and a
 * client id (api.clientId or api.initialClientId). Returns {run, cancel}.
 *  run(prompt, helperId, onStatus) -> Promise of the helper's payload (ui.c2c_viewer[0]).
 *  onStatus(state, text): "queued" | "waiting" | "running".
 */
export function createRunner(api, { pollMs = 1000, vanishPolls = 6, setTimer = setTimeout, clearTimer = clearTimeout } = {}) {
    let active = null;

    const clientId = () => api.clientId ?? api.initialClientId ?? null;

    async function json(route, init) {
        const r = await api.fetchApi(route, init);
        let body = null;
        try { body = await r.json(); } catch (_e) { body = null; }
        return { ok: r.ok, status: r.status, body };
    }

    function run(prompt, helperId, onStatus = () => {}) {
        if (active) return Promise.reject(new Error("The viewer is already waiting for a job."));
        return new Promise((resolve, reject) => {
            const buffer = [];
            let pid = null;
            let done = false;
            let timer = null;
            let misses = 0;
            const types = ["executed", "execution_error", "execution_interrupted", "execution_start", "executing"];

            const finish = (err, payload) => {
                if (done) return;
                done = true;
                active = null;
                if (timer) clearTimer(timer);
                for (const t of types) api.removeEventListener(t, onEvent);
                if (err) reject(err); else resolve(payload);
            };
            const handle = (type, d) => {
                if (!d || d.prompt_id !== pid) return;
                if (type === "executed" && String(d.node ?? d.display_node) === helperId) {
                    const p = d.output?.[UI_KEY]?.[0];
                    if (p) finish(null, p); else finish(new Error("The viewer's helper ran but returned nothing."));
                } else if (type === "execution_error") {
                    finish(new Error(`${d.node_type || "A node"} failed: ${d.exception_message || "unknown error"}`.trim()));
                } else if (type === "execution_interrupted") {
                    finish(new Error("Stopped."));
                } else if (type === "execution_start" || type === "executing") {
                    onStatus("running", "Running…");
                }
            };
            function onEvent(e) {
                const d = e?.detail;
                if (pid === null) { buffer.push([e.type, d]); return; }
                handle(e.type, d);
            }
            for (const t of types) api.addEventListener(t, onEvent);

            const poll = async () => {
                timer = null;
                if (done) return;
                try {
                    const h = await json(`/history/${encodeURIComponent(pid)}`);
                    if (done) return;
                    const entry = h.body?.[pid];
                    if (entry) {
                        const out = entry.outputs?.[helperId]?.[UI_KEY]?.[0];
                        if (out) return finish(null, out);
                        const st = entry.status;
                        if (st?.status_str === "error") {
                            const msg = (st.messages || []).find((m) => m[0] === "execution_error")?.[1];
                            return finish(new Error(`${msg?.node_type || "A node"} failed: ${msg?.exception_message || "see the ComfyUI console"}`));
                        }
                        if (st?.completed) return finish(new Error("The viewer's helper finished without an answer."));
                    }
                    const q = await json("/queue");
                    if (done) return;
                    const running = q.body?.queue_running || [], pending = q.body?.queue_pending || [];
                    const has = (list) => list.some((it) => it?.[1] === pid);
                    if (has(pending)) {
                        misses = 0;
                        onStatus("waiting", running.length ? "Waiting for ComfyUI: another job is running." : "Queued.");
                    } else if (has(running)) {
                        misses = 0;
                        onStatus("running", "Running…");
                    } else if (!entry && (misses += 1) >= vanishPolls) {
                        return finish(new Error("The viewer's job disappeared from ComfyUI (was the server restarted?). Try again."));
                    }
                } catch (_e) {
                    if ((misses += 1) >= vanishPolls) {
                        return finish(new Error("ComfyUI does not answer (was the server stopped?). Try again when it is back."));
                    }
                }
                if (!done) timer = setTimer(poll, pollMs);
            };

            active = { cancel: async () => {
                if (done) return;
                if (pid) {
                    try { await api.fetchApi("/queue", { method: "POST", body: JSON.stringify({ delete: [pid] }) }); } catch (_e) { /* ignore */ }
                    try { await api.fetchApi("/interrupt", { method: "POST", body: JSON.stringify({ prompt_id: pid }) }); } catch (_e) { /* ignore */ }
                }
                finish(new Error("Stopped."));
            } };

            (async () => {
                try {
                    const r = await json("/prompt", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ client_id: clientId(), prompt, front: true, extra_data: { c2c_viewer: true } }),
                    });
                    if (done) return;
                    if (!r.ok || !r.body?.prompt_id) return finish(new Error(validationMessage(r.body)));
                    pid = r.body.prompt_id;
                    onStatus("queued", "Queued.");
                    for (const [t, d] of buffer.splice(0)) handle(t, d);
                    if (!done) timer = setTimer(poll, pollMs);
                } catch (e) {
                    finish(new Error(`Could not reach ComfyUI: ${e?.message || e}`));
                }
            })();
        });
    }

    return {
        run,
        cancel: () => active?.cancel(),
        get busy() { return !!active; },
    };
}

/** The /view URL of a temp file the helpers wrote. */
export function viewUrl(api, ref) {
    const q = new URLSearchParams({ filename: ref.filename, subfolder: ref.subfolder || "", type: ref.type || "temp" });
    return api.apiURL ? api.apiURL(`/view?${q}`) : `/view?${q}`;
}
