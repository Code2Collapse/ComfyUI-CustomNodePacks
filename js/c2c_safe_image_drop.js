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

/** @type {"auto"|"image_only"|"comfy_default"} */
let _mode = "auto";
let _enabled = true;

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

        const assessment = await _assessFile(file);

        // Optional metadata inspector modal (valid workflow only).
        if (assessment.status === "valid" && app.__c2cMetaInspectPrompt) {
            try {
                const proceed = await app.__c2cMetaInspectPrompt(assessment.meta, assessment, file);
                if (proceed === false) return loadImageOnly(file, evt);
                if (proceed === "workflow" && assessment.kind === "comfy-workflow" && app.loadGraphData) {
                    try {
                        await app.loadGraphData(assessment.data);
                        return;
                    } catch (e) {
                        toast(`Workflow load failed — loaded image only (${file.name})`, "warning");
                        return loadImageOnly(file, evt);
                    }
                }
            } catch (_) { /* fall through to Comfy */ }
        }

        if (assessment.status === "corrupt") {
            toast(
                `Workflow metadata in ${file.name} is damaged — loaded image only.`,
                "warning",
            );
            return loadImageOnly(file, evt);
        }

        if (_mode === "image_only" && assessment.status === "none") {
            return loadImageOnly(file, evt);
        }

        // Valid workflow or plain image: ComfyUI default.
        try {
            return await orig(file, ...args);
        } catch (e) {
            if (_isImageFile(file)) {
                console.warn("[C2C.SafeDrop] Comfy handleFile failed, falling back to image-only:", e);
                toast(`Drop failed — loaded image only (${file.name})`, "warning");
                return loadImageOnly(file, evt);
            }
            throw e;
        }
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
        _enabled = app.ui.settings.getSettingValue(SETTING_ENABLED, true);
        _mode = app.ui.settings.getSettingValue(SETTING_MODE, "auto");
        _installHandleFileWrap();
    },
});
