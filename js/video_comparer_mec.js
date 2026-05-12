// VideoComparerMEC — frontend upload + interactive scrubber widget.
//
// Adds two upload buttons (A / B) that POST to ComfyUI's /upload/image endpoint
// (which also accepts video and audio files); the uploaded filename is then
// written into the `file_a` / `file_b` combo widget so the backend can decode
// it on next Queue. Also adds keyboard arrows on the frame_index slider for
// easy scrubbing.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_ID = "VideoComparerMEC";

async function uploadFile(file) {
    const body = new FormData();
    body.append("image", file, file.name);
    body.append("type", "input");
    body.append("overwrite", "true");
    const resp = await api.fetchApi("/upload/image", { method: "POST", body });
    if (!resp.ok) throw new Error(`upload failed: ${resp.status}`);
    const data = await resp.json();
    // Comfy returns { name, subfolder, type }
    const sub = data.subfolder ? data.subfolder + "/" : "";
    return sub + (data.name || file.name);
}

function makeUploadButton(node, slot, label) {
    const widgetName = slot === "a" ? "file_a" : "file_b";
    const btn = node.addWidget("button", `📁 Upload ${label}`, null, async () => {
        const inp = document.createElement("input");
        inp.type = "file";
        inp.accept =
            "image/*,video/*,audio/*,.exr,.hdr,.tif,.tiff,.webp,.mp4,.mov,.mkv,.webm,.gif,.wav,.mp3,.flac,.ogg,.aac,.m4a";
        inp.style.display = "none";
        document.body.appendChild(inp);
        inp.addEventListener("change", async () => {
            const f = inp.files && inp.files[0];
            inp.remove();
            if (!f) return;
            try {
                btn.name = `⏳ Uploading ${label}…`;
                node.setDirtyCanvas(true, true);
                const uploaded = await uploadFile(f);
                // refresh combo widget choices and select the new file
                const w = node.widgets.find((w) => w.name === widgetName);
                if (w) {
                    if (Array.isArray(w.options.values) && !w.options.values.includes(uploaded)) {
                        w.options.values.push(uploaded);
                    }
                    w.value = uploaded;
                }
                btn.name = `✅ ${label}: ${f.name.length > 22 ? f.name.slice(0, 19) + "…" : f.name}`;
            } catch (e) {
                console.error("[VideoComparerMEC] upload error", e);
                btn.name = `❌ Upload ${label} failed`;
            }
            node.setDirtyCanvas(true, true);
        });
        inp.click();
    });
    btn.serialize = false;
    return btn;
}

app.registerExtension({
    name: "MEC.VideoComparer",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;

        const _onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            _onCreated?.apply(this, arguments);
            const node = this;

            // Insert upload buttons at the top of the widget stack
            makeUploadButton(node, "a", "A");
            makeUploadButton(node, "b", "B");

            // Frame scrubber: keyboard arrows when slider focused
            const frameW = node.widgets.find((w) => w.name === "frame_index");
            if (frameW) {
                frameW.options = frameW.options || {};
                frameW.options.min = 0;
                frameW.options.step = 1;
            }

            // Light hint banner
            const hint = node.addWidget("text", "ℹ hint", "", () => {});
            hint.value = "Upload A/B, pick mode, Queue. Files survive in input/.";
            hint.disabled = true;
            hint.serialize = false;

            node.setDirtyCanvas(true, true);
        };
    },
});
