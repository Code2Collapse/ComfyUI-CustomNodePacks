// c2c_link_snap.js — a wire released anywhere on a node connects to the matching slot (L2.39, owner A9).
// ---------------------------------------------------------------------------
// Owner: "i drag a wire from output to another node and it auto-attaches to its connection ... when the pointer
// is near to the node". Measured on frontend 1.52.7 (docs/evidence/L2.38/snap_all_c2c.*.json): every widget row is
// also an input socket, so with the pointer over a widget row core targets THAT widget's input. When the wire's
// type does not fit it (an IMAGE wire over a FLOAT slider) there is no snap preview and the release does nothing.
// It still works over the title, gaps and previews, so on widget-heavy nodes (most of ours) it looks broken. Same
// result with every C2C front-end file blocked: this is core behaviour, not damage, but it is what the owner wants.
//
// With "Wire drop snaps to the matching slot" on (default):
//   - release: if the slot under the pointer cannot take the wire, connect to the node's first matching slot
//     (core's own connectToNode, the path it already takes over the title or a gap);
//   - hover: if core found no snap target on a node that can take the wire, preview that first matching slot.
// A drop on a slot that fits is unchanged (a compatible widget input still takes the wire). Off = core exactly.

import { app } from "../../scripts/app.js";
import { getRuntime } from "./_c2c_runtime.js";

const SETTING_ID = "c2c.linkSnap.matchingSlot";
let _enabled = true;

/** The input (or output) slot core would drop on at this canvas position, as in LinkConnector.dropOnNode. */
function slotUnderPointer(lc, node, x, y) {
    if (lc.state?.connectingTo === "output") return node.getOutputOnPos?.([x, y]);
    return node.getInputOnPos?.([x, y]) ?? node.getSlotFromWidget?.(lc.overWidget);
}

function fits(lc, node, slot) {
    return lc.state?.connectingTo === "output"
        ? lc.renderLinks.some((l) => l.canConnectToOutput?.(node, slot))
        : lc.isInputValidDrop(node, slot);
}

function patchDrop(lc) {
    const proto = Object.getPrototypeOf(lc);
    getRuntime().safePatch(proto, "dropOnNode", (orig) => function (node, event) {
        if (_enabled && node && event && !this.renderLinks.every((l) => l.node === node)) {
            const slot = slotUnderPointer(this, node, event.canvasX, event.canvasY);
            if (slot && !fits(this, node, slot)) return this.connectToNode(node, event);
        }
        return orig.call(this, node, event);
    }, { id: "linksnap.dropOnNode" });
}

/** Core binds processMouseMove to the canvas element once, at canvas creation, so a prototype patch never runs.
 *  A later pointermove listener on the same element runs right after core's, in the same event, with core's
 *  hover state (node_over, _highlight_pos) already computed for this pointer position. */
function installPreview(canvas) {
    const el = canvas?.canvas;
    if (!el || el.__c2cLinkSnap) return;
    el.__c2cLinkSnap = true;
    el.addEventListener("pointermove", getRuntime().guard(() => {
        if (!_enabled) return;
        const lc = canvas.linkConnector;
        const node = canvas.node_over;
        if (!lc?.isConnecting || !node || canvas._highlight_pos || lc.state?.connectingTo !== "input") return;
        const link = lc.renderLinks?.[0];
        if (!link || link.node === node || !lc.isNodeValidDrop(node)) return;
        const found = node.findInputByType?.(link.fromSlot?.type);
        if (!found?.slot || typeof node.getInputSlotPos !== "function") return;
        canvas._highlight_pos = node.getInputSlotPos(found.slot);
        canvas._highlight_input = found.slot;
        canvas.dirty_canvas = true;
    }, "linksnap:preview"), { passive: true });
}

app.registerExtension({
    name: "C2C.LinkSnap",
    settings: [{
        id: SETTING_ID,
        name: "Wire drop snaps to the matching slot",
        tooltip: "Release a wire anywhere on a node (also over a widget) and it connects to the first slot of the "
            + "matching type, with the snap preview while you hover. ComfyUI 1.52 otherwise targets the widget "
            + "under the pointer and does nothing when the type does not fit. Off = ComfyUI's own behaviour.",
        type: "boolean",
        defaultValue: true,
        category: ["c2c", "Canvas", "Link snapping"],
        onChange: (v) => { _enabled = v !== false; },
    }],
    async setup() {
        _enabled = app.ui.settings.getSettingValue(SETTING_ID) !== false;
        const lc = app.canvas?.linkConnector;
        if (lc) patchDrop(lc);
        installPreview(app.canvas);
    },
});
