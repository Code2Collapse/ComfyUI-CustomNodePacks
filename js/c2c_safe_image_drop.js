// c2c_safe_image_drop.js — corrupt/missing workflow metadata → image-only load
// Wraps app.handleFile (ComfyUI native path). Does NOT capture-phase block drops.
import { app } from "../../scripts/app.js";
import {
    readPngTextChunks,
    assessWorkflowMetadata,
    loadImageOnly,
    toast,
} from "./c2c_png_metadata.js";

const SETTING_MODE = "c2c.safeDrop.mode";
const SETTING_ENABLED = "c2c.safeDrop.enabled";
const SETTING_CONFIRM = "c2c.safeDrop.confirmWorkflow";

/** @type {"auto"|"image_only"|"comfy_default"} */
let _mode = "auto";
let _enabled = true;
let _confirmWorkflow = false;   // default: auto-load embedded workflow, no modal

/** True if the drop landed on top of an existing node that takes an image
 *  (e.g. a LoadImage). In that case we must NOT create a new node — let
 *  ComfyUI's native path upload into the hovered node. */
function _dropOnImageNode(evt) {
    try {
        const canvas = app.canvas;
        if (!evt || !canvas?.canvas || !app.graph?._nodes) return false;
        const rect = canvas.canvas.getBoundingClientRect();
        const ds = canvas.ds;
        const gx = (evt.clientX - rect.left - ds.offset[0]) / ds.scale;
        const gy = (evt.clientY - rect.top  - ds.offset[1]) / ds.scale;
        for (let i = app.graph._nodes.length - 1; i >= 0; i--) {
            const n = app.graph._nodes[i];
            const [nx, ny] = n.pos || [0, 0];
            const [nw, nh] = n.size || [0, 0];
            if (gx >= nx && gx <= nx + nw && gy >= ny - 30 && gy <= ny + nh) {
                return (n.widgets || []).some((w) => w?.name === "image") ||
                       /loadimage|image/i.test(n.type || "");
            }
        }
    } catch (_) { /* fall through */ }
    return false;
}

function _isImageFile(file) {
    if (!file) return false;
    const n = (file.name || "").toLowerCase();
    const t = file.type || "";
    return t.startsWith("image/") || /\.(png|jpe?g|webp|gif|bmp)$/i.test(n);
}

async function _assessFile(file) {
    if (!file || !((file.name || "").toLowerCase().endsWith(".png") || file.type === "image/png")) {
        return { status: "none", meta: {} };
    }
    try {
        const meta = readPngTextChunks(await file.arrayBuffer());
        return { ...assessWorkflowMetadata(meta), meta };
    } catch (e) {
        return { status: "corrupt", error: String(e?.message || e), meta: {} };
    }
}

function _installHandleFileWrap() {
    if (app.__c2cSafeDropInstalled) return;
    app.__c2cSafeDropInstalled = true;
    app.__c2cOrigHandleFile = app.handleFile?.bind(app);

    document.addEventListener("drop", (e) => {
        app.__c2cLastDropEvent = e;
    }, true);

    app.handleFile = async function c2cSafeHandleFile(file, ...args) {
        const orig = app.__c2cOrigHandleFile;
        if (!_enabled || !_isImageFile(file) || !orig) {
            return orig?.(file, ...args);
        }

        // Shift = user explicitly wants ComfyUI default (workflow load attempt).
        const evt = app.__c2cLastDropEvent;
        if (evt?.shiftKey) return orig(file, ...args);

        // Alt = force image-only (skip workflow even if valid).
        if (evt?.altKey) return loadImageOnly(file, evt);

        if (_mode === "comfy_default") return orig(file, ...args);

        // Dropped ON an existing image node → let native upload into it.
        if (_dropOnImageNode(evt)) return orig(file, ...args);

        const assessment = await _assessFile(file);

        // ① Image carries a valid ComfyUI workflow → AUTO-load it (no modal,
        //    unless the user explicitly opted into the confirm dialog).
        if (assessment.status === "valid" && assessment.kind === "comfy-workflow" && app.loadGraphData) {
            if (_confirmWorkflow && app.__c2cMetaInspectPrompt) {
                try {
                    const proceed = await app.__c2cMetaInspectPrompt(assessment.meta, assessment, file);
                    if (proceed === false) return loadImageOnly(file, evt);
                } catch (_) { /* dialog failed — fall through to auto-load */ }
            }
            try {
                await app.loadGraphData(assessment.data);
                toast(`Loaded workflow from ${file.name}`, "info");
                return;
            } catch (e) {
                toast(`Workflow load failed — loaded image only (${file.name})`, "warning");
                return loadImageOnly(file, evt);
            }
        }

        // ② a1111 / prompt-only metadata → let ComfyUI try its parsers.
        if (assessment.status === "valid") {
            try { return await orig(file, ...args); }
            catch (e) { return loadImageOnly(file, evt); }
        }

        // ③ Damaged workflow metadata → image only.
        if (assessment.status === "corrupt") {
            toast(`Workflow metadata in ${file.name} is damaged — loaded image only.`, "warning");
            return loadImageOnly(file, evt);
        }

        // ④ Plain image (no metadata) dropped on empty canvas → GUARANTEE a
        //    LoadImage node with the image at the drop point. (Previously this
        //    deferred to native handleFile, which often did nothing.)
        return loadImageOnly(file, evt);
    };
}

app.registerExtension({
    name: "C2C.SafeImageDrop",
    async setup() {
        app.ui.settings.addSetting({
            id: SETTING_ENABLED,
            name: "C2C ▸ Safe image drop (corrupt metadata fallback)",
            type: "boolean",
            defaultValue: true,
            onChange: (v) => { _enabled = !!v; },
        });
        app.ui.settings.addSetting({
            id: SETTING_MODE,
            name: "C2C ▸ Safe drop mode",
            type: "combo",
            defaultValue: "auto",
            options: [
                { value: "auto", text: "Auto — Comfy default; corrupt → image only" },
                { value: "image_only", text: "Image only when no workflow metadata" },
                { value: "comfy_default", text: "Comfy default (disable safe fallback)" },
            ],
            onChange: (v) => { _mode = v || "auto"; },
        });
        app.ui.settings.addSetting({
            id: SETTING_CONFIRM,
            name: "C2C ▸ Confirm before loading embedded workflow",
            tooltip: "Off (default): dropping an image with an embedded ComfyUI workflow loads it automatically. On: show the metadata inspector first.",
            type: "boolean",
            defaultValue: false,
            onChange: (v) => { _confirmWorkflow = !!v; },
        });
        _enabled = app.ui.settings.getSettingValue(SETTING_ENABLED, true);
        _mode = app.ui.settings.getSettingValue(SETTING_MODE, "auto");
        _confirmWorkflow = app.ui.settings.getSettingValue(SETTING_CONFIRM, false);
        _installHandleFileWrap();
    },
});
