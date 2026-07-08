// c2c_ai_workflow_builder.js — "Build workflow from description" (AI-Spine).
//
// Palette command + body-mounted dialog (never inside a node: third-party
// overlays eclipse in-node popovers — see the rgthree-progress-bar incident).
// POSTs /c2c/ai/build_workflow; on success asks before REPLACING the canvas
// (loadGraphData is whole-graph), then loads the returned LiteGraph JSON.

import { app } from "../../scripts/app.js";

function openBuilderDialog() {
    const old = document.getElementById("__c2c_wf_builder");
    if (old) { old.remove(); return; }
    const wrap = document.createElement("div");
    wrap.id = "__c2c_wf_builder";
    wrap.style.cssText = `
        position:fixed;inset:0;z-index:2147483000;display:flex;
        align-items:center;justify-content:center;background:rgba(0,0,0,0.55);
    `;
    const panel = document.createElement("div");
    panel.style.cssText = `
        width:min(640px,92vw);background:var(--c2c-bg, #1e1e2e);color:var(--c2c-fg, #cdd6f4);
        border:1px solid var(--c2c-surface1, #45475a);border-radius:10px;padding:16px;
        font:13px system-ui,sans-serif;box-shadow:0 12px 48px rgba(0,0,0,0.6);
    `;
    panel.innerHTML = `
        <div style="font-size:15px;font-weight:600;margin-bottom:4px;">🪄 Build a workflow from a description</div>
        <div style="color:var(--c2c-overlay1,#7f849c);margin-bottom:10px;">
            Describe what you want. The agent designs a graph using this workspace's
            nodes, validates it against the live registry, and loads it here.
        </div>
        <textarea rows="3" placeholder="e.g. basic text to image with SDXL at 1024, 30 steps"
            style="width:100%;box-sizing:border-box;resize:vertical;background:var(--c2c-scrimDark3,#11111b);
                   color:inherit;border:1px solid var(--c2c-surface1,#45475a);border-radius:6px;padding:8px;"></textarea>
        <div data-status style="min-height:20px;margin:8px 0;color:var(--c2c-overlay1,#7f849c);"></div>
        <div style="display:flex;gap:8px;justify-content:flex-end;">
            <button data-cancel style="padding:6px 14px;border-radius:6px;border:1px solid var(--c2c-surface1,#45475a);
                background:transparent;color:inherit;cursor:pointer;">Cancel</button>
            <button data-go style="padding:6px 14px;border-radius:6px;border:1px solid var(--c2c-blue,#89b4fa);
                background:var(--c2c-blue,#89b4fa);color:#11111b;font-weight:600;cursor:pointer;">Build</button>
        </div>
    `;
    wrap.appendChild(panel);
    document.body.appendChild(wrap);
    const ta = panel.querySelector("textarea");
    const status = panel.querySelector("[data-status]");
    const close = () => wrap.remove();
    panel.querySelector("[data-cancel]").onclick = close;
    wrap.addEventListener("pointerdown", (e) => { if (e.target === wrap) close(); });
    ta.focus();

    panel.querySelector("[data-go]").onclick = async () => {
        const req = ta.value.trim();
        if (!req) { status.textContent = "Describe the workflow first."; return; }
        status.textContent = "Designing the graph… (local models can take ~a minute)";
        let r = null;
        try {
            r = await (await fetch("/c2c/ai/build_workflow", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ request: req }),
            })).json();
        } catch (e) {
            status.textContent = "Request failed: " + e;
            return;
        }
        if (!r?.ok) {
            const det = (r?.details || []).slice(0, 3).join(" · ");
            status.textContent = "Could not build: " + (r?.error || "unknown") + (det ? " — " + det : "");
            return;
        }
        const n = r.node_count, via = r.provider;
        if (!confirm(`Built a ${n}-node workflow (via ${via}). Replace the current canvas with it?`)) {
            status.textContent = "Kept your current canvas. The plan is in the console.";
            console.log("[C2C] workflow plan", r.plan);
            return;
        }
        close();
        await app.loadGraphData(r.graph);
        app.extensionManager?.toast?.add?.({
            severity: "success", summary: "Workflow built",
            detail: `${n} nodes via ${via}`, life: 5000,
        });
    };
}

app.registerExtension({
    name: "C2C.AIWorkflowBuilder",
    commands: [{
        id: "c2c.ai.buildWorkflow",
        label: "C2C: Build workflow from description (AI)",
        icon: "pi pi-sparkles",
        function: openBuilderDialog,
    }],
    keybindings: [{
        combo: { key: "b", ctrl: true, alt: true },
        commandId: "c2c.ai.buildWorkflow",
    }],
});

// Programmatic + test entry point.
window.__c2cBuildWorkflow = openBuilderDialog;
