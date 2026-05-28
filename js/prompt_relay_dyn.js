// Prompt Relay Encode — dynamic socket + widget visibility.
//
// The merged PromptRelayEncodeC2C node declares ALL possible sockets so a
// single node can serve every backend (native / smart / kijai). This
// extension hides the inputs, outputs, and widgets that do not apply to the
// currently-selected `backend` dropdown, keeping the UI clean while leaving
// the Python validator happy (every backend-specific input is optional).
//
// Hook: app.registerExtension(beforeRegisterNodeDef) — when the node class
// matches PromptRelayEncodeC2C, wrap onNodeCreated and the backend widget's
// callback to call applyBackend(node).

import { app } from "../../scripts/app.js";
import { setWidgetVisible } from "./_widget_visibility.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";

const NODE_NAME = "PromptRelayEncodeC2C";

// Backend → {inputs, widgets, outputs} visible-set definitions.
// Anything not listed is hidden. "global_prompt", "epsilon", "backend" and
// "relay_options" are universal.
const COMMON_INPUTS  = new Set(["relay_options"]);
const COMMON_WIDGETS = new Set(["backend", "global_prompt", "epsilon"]);

const BACKEND_INPUTS = {
    native: new Set(["model", "clip", "latent"]),
    smart:  new Set(["model", "clip", "latent"]),
    kijai:  new Set(["wan_model", "wan_t5"]),
};

const BACKEND_WIDGETS = {
    native: new Set(["local_prompts", "segment_lengths"]),
    smart:  new Set(["smart_prompt", "normalize_by_tokens"]),
    kijai:  new Set(["local_prompts", "segment_lengths", "latent_frames",
                     "negative_prompt", "encode_device"]),
};

// Output names that should be visible per backend.
const BACKEND_OUTPUTS = {
    native: new Set(["model", "positive"]),
    smart:  new Set(["model", "positive"]),
    kijai:  new Set(["wan_model", "wan_text_embeds"]),
};

function setInputVisible(node, idx, visible) {
    const slot = node.inputs?.[idx];
    if (!slot) return;
    if (visible) {
        if ("__pr_origType" in slot) {
            slot.type = slot.__pr_origType;
            delete slot.__pr_origType;
        }
    } else {
        // Disconnect any wired link before hiding.
        if (slot.link != null && node.graph) {
            try { node.graph.removeLink(slot.link); } catch (__c2cErr) { __c2cReport("prompt_relay_dyn", __c2cErr); }
        }
        if (!("__pr_origType" in slot)) slot.__pr_origType = slot.type;
        // LiteGraph hides slots whose type starts with "-".
        slot.type = "-" + (slot.__pr_origType || "");
    }
}

function setOutputVisible(node, idx, visible) {
    const slot = node.outputs?.[idx];
    if (!slot) return;
    if (visible) {
        if ("__pr_origType" in slot) {
            slot.type = slot.__pr_origType;
            delete slot.__pr_origType;
        }
    } else {
        if (Array.isArray(slot.links) && slot.links.length && node.graph) {
            for (const linkId of [...slot.links]) {
                try { node.graph.removeLink(linkId); } catch (__c2cErr) { __c2cReport("prompt_relay_dyn", __c2cErr); }
            }
        }
        if (!("__pr_origType" in slot)) slot.__pr_origType = slot.type;
        slot.type = "-" + (slot.__pr_origType || "");
    }
}

function applyBackend(node) {
    const w = node.widgets?.find((x) => x.name === "backend");
    const backend = w?.value || "native";
    const visIns = BACKEND_INPUTS[backend] || new Set();
    const visOuts = BACKEND_OUTPUTS[backend] || new Set();
    const visWidgets = BACKEND_WIDGETS[backend] || new Set();

    if (Array.isArray(node.inputs)) {
        node.inputs.forEach((slot, idx) => {
            const keep = COMMON_INPUTS.has(slot.name) || visIns.has(slot.name);
            setInputVisible(node, idx, keep);
        });
    }
    if (Array.isArray(node.outputs)) {
        node.outputs.forEach((slot, idx) => {
            setOutputVisible(node, idx, visOuts.has(slot.name));
        });
    }
    if (Array.isArray(node.widgets)) {
        for (const widget of node.widgets) {
            const keep = COMMON_WIDGETS.has(widget.name) || visWidgets.has(widget.name);
            setWidgetVisible(widget, keep);
        }
    }
    node.setSize?.(node.computeSize?.() || node.size);
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "c2c.prompt_relay.dynamic_sockets",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated?.apply(this, arguments);
            // Defer once so widgets/inputs are fully populated by ComfyUI.
            queueMicrotask(() => {
                const w = this.widgets?.find((x) => x.name === "backend");
                if (w) {
                    const origCb = w.callback;
                    w.callback = (value) => {
                        const ret = origCb?.call(w, value);
                        applyBackend(this);
                        return ret;
                    };
                }
                applyBackend(this);
            });
            return r;
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (data) {
            const r = origOnConfigure?.apply(this, arguments);
            queueMicrotask(() => applyBackend(this));
            return r;
        };
    },
});
