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

const NODE_NAME = "SplinePathFlowMaskMEC";

app.registerExtension({
    name: "MEC.SplinePathFlowMask.EmbeddedEditor",

    async nodeCreated(node) {
        if (!node || node.comfyClass !== NODE_NAME) return;

        // Wait one tick so all widgets are realized.
        await new Promise((r) => setTimeout(r, 0));

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

        if (wantEditor) {
            try {
                ed.installEditor(node, { hostNode: NODE_NAME });
            } catch (err) {
                console.error("[SplinePathFlowMask] installEditor failed:", err);
            }
        }

        // React to the toggle: if user flips it after creation, reload the
        // graph or hint. We avoid hot-swapping the canvas to keep state simple.
        if (useEditorW && useEditorW.callback === undefined) {
            useEditorW.callback = (v) => {
                if (v && !node.__mec_editor_installed) {
                    try {
                        ed.installEditor(node, { hostNode: NODE_NAME });
                    } catch (e) {
                        console.error(e);
                    }
                }
                // Hiding the canvas mid-session is not supported; instruct
                // user to reload the workflow to fully remove it.
                node.setDirtyCanvas?.(true, true);
            };
        }
    },
});
