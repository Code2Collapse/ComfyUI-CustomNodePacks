// =====================================================================
// SplinePathFlowMaskMEC frontend extension
// =====================================================================
// Adds an OPTIONAL embedded spline editor inside the SplinePathFlowMask
// node, sharing the same canvas widget code as SplineMaskEditorMEC.
//
// Strategy:
//   - The Python node exposes a STRING widget `spline_data` and a BOOLEAN
//     widget `use_embedded_editor`.
//   - At node creation we check `use_embedded_editor`; if true we call
//     `window.__MEC_SPLINE_EDITOR__.installEditor(node)` to attach the
//     same DOM canvas + tool palette used by SplineMaskEditorMEC.
//   - Toggling `use_embedded_editor` later collapses / restores the
//     canvas (set widget computeSize -> 0 to hide).
//
// No HTTP / fetch calls. All DOM. Compatible with comfyui_qa_stress_test
// v2 rules and reuses the SplineMaskEditor module (which itself respects
// the centripetal_alpha widget for smooth no-cusp curves).
// =====================================================================

import { app } from "/scripts/app.js";
import { installModeGated } from "./_mode_gate.js";

// Targets unified SplineMaskMEC (mode=flow_path) and any legacy
// SplinePathFlowMaskMEC nodes on saved graphs.
const NODE_NAMES = ["SplineMaskMEC", "SplinePathFlowMaskMEC"];
const NODE_NAME = "SplineMaskMEC";

function _installFlowEditor(node) {
    const ed = window.__MEC_SPLINE_EDITOR__;
    if (!ed || typeof ed.installEditor !== "function") {
        console.warn(
            "[SplinePathFlowMask] SplineMaskEditor module not loaded; " +
            "embedded editor unavailable. Use the raw spline_data widget."
        );
        return;
    }
    const useEditorW = node.widgets?.find?.((w) => w.name === "use_embedded_editor");
    const wantEditor = useEditorW ? !!useEditorW.value : true;
    if (!wantEditor) return;
    try {
        ed.installEditor(node, { hostNode: NODE_NAME });
        node.__mec_editor_installed = true;
    } catch (err) {
        console.error("[SplinePathFlowMask] installEditor failed:", err);
    }
}

app.registerExtension({
    name: "MEC.SplinePathFlowMask.EmbeddedEditor",

    async nodeCreated(node) {
        if (!node || !NODE_NAMES.includes(node.comfyClass)) return;
        await new Promise((r) => setTimeout(r, 0));

        if (node.comfyClass === "SplineMaskMEC") {
            installModeGated(node, {
                activeWhen: "flow_path",
                installerKey: "splineFlowPath",
                installer: (n) => _installFlowEditor(n),
                hostFinder: (n) => n._mecSplineEditHost || null,
                widgetFinder: (n) => n._mecSplineEditWidget || null,
                widgetHeight: (n) => (typeof n._mecSplineEditWidgetH === "function" ? n._mecSplineEditWidgetH() : 460),
            });
            return;
        }

        // Legacy direct binding.
        _installFlowEditor(node);
        const useEditorW = node.widgets?.find?.((w) => w.name === "use_embedded_editor");
        if (useEditorW && useEditorW.callback === undefined) {
            useEditorW.callback = (v) => {
                if (v && !node.__mec_editor_installed) {
                    _installFlowEditor(node);
                }
                node.setDirtyCanvas?.(true, true);
            };
        }
    },
});
