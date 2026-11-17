// Previews that come back after a workflow-tab switch or a page reload (L9.282 P1.4 / L9.169).
//
// Verified in the installed frontend 1.53.10 bundle (settingStore / core chunks):
// - a tab switch runs ChangeTracker.restore() -> nodeOutputStore.restoreOutputs(saved), which refills
//   app.nodeOutputs keyed by NODE LOCATOR id ("<nodeId>" in the root graph, "<subgraphId>:<nodeId>" in a subgraph)
//   but calls neither the nodes' onExecuted nor any extension hook - so a node that draws its own preview from
//   onExecuted came back empty;
// - queue-history loads call the extension hook onNodeOutputsUpdated(outputs) keyed by EXECUTION id;
// - node.hideOutputImages = true stops core drawing its own image preview under a node (core's ImageCrop /
//   Compositor use it).
// replayLastOutput(node) feeds the node's stored output back into its own onExecuted once per distinct output.

function appRef() {
    return window.comfyAPI?.app?.app ?? window.app ?? null;
}

export function locatorId(node) {
    const g = node?.graph;
    const root = appRef()?.rootGraph ?? appRef()?.graph;
    const inSub = g && root && g !== root && g.id && g.isRootGraph !== true;
    return inSub ? `${g.id}:${node.id}` : String(node?.id);
}

function signature(out) {
    try {
        return JSON.stringify(out).slice(0, 4096);
    } catch (_e) {
        return String(Math.random());
    }
}

/** Replay the node's last stored output into its onExecuted (no-op when nothing is stored or it was already shown). */
export function replayLastOutput(node, output) {
    const a = appRef();
    const out = output ?? a?.nodeOutputs?.[locatorId(node)];
    if (!out || typeof node?.onExecuted !== "function") return false;
    const sig = signature(out);
    if (node.__c2cShownSig === sig) return false;
    node.__c2cShownSig = sig;
    try {
        node.onExecuted(out);
    } catch (e) {
        console.warn("[C2C] preview restore failed:", e);
        return false;
    }
    return true;
}

/**
 * For a node type that draws its own preview: remember what it showed, replay the stored output after a tab switch /
 * reload, and (hideCore) stop core's duplicate image preview under it. Call from beforeRegisterNodeDef.
 */
const _types = new Set();
let _extRegistered = false;

export function installPreviewRestore(nodeType, { hideCore = true, type = null } = {}) {
    const proto = nodeType.prototype;
    if (proto.__c2cPreviewRestore) return;
    proto.__c2cPreviewRestore = true;
    const t = type ?? nodeType.comfyClass ?? nodeType.type;
    if (t) _types.add(t);
    if (!_extRegistered) {                              // one history hook for every adopted type
        const a = appRef();
        if (a?.registerExtension) {
            _extRegistered = true;
            a.registerExtension(previewRestoreExtension(_types));
        }
    }
    const onExecuted = proto.onExecuted;
    proto.onExecuted = function (out) {
        this.__c2cShownSig = signature(out);            // a live result: remember it so the replay does not repeat it
        return onExecuted?.apply(this, arguments);
    };
    const onNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
        const r = onNodeCreated?.apply(this, arguments);
        if (hideCore) this.hideOutputImages = true;
        return r;
    };
    const onConfigure = proto.onConfigure;
    proto.onConfigure = function () {
        const r = onConfigure?.apply(this, arguments);
        if (hideCore) this.hideOutputImages = true;
        // the store is refilled right after the graph loads: look again once the current task is done
        setTimeout(() => replayLastOutput(this), 0);
        setTimeout(() => replayLastOutput(this), 300);
        return r;
    };
}

/** One extension hook for every adopted type: queue-history loads (keys are execution ids). */
export function previewRestoreExtension(types) {
    const want = types instanceof Set ? types : new Set(types);     // a live Set: types adopted later count too
    return {
        name: "C2C.PreviewRestore",
        onNodeOutputsUpdated(outputs) {
            const a = appRef();
            if (!a?.rootGraph && !a?.graph) return;
            for (const [execId, out] of Object.entries(outputs || {})) {
                const parts = String(execId).split(":");
                let g = a.rootGraph ?? a.graph;
                let node = null;
                for (let i = 0; i < parts.length && g; i++) {
                    node = g.getNodeById?.(Number(parts[i])) ?? g.getNodeById?.(parts[i]);
                    g = i < parts.length - 1 ? node?.subgraph : null;
                }
                if (node && want.has(node.comfyClass ?? node.type)) replayLastOutput(node, out);
            }
        },
    };
}
