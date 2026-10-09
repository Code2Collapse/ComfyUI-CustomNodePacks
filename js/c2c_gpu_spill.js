/**
 * GPU spill notice (L7.59): when a run used more GPU memory than the card has, Windows' NVIDIA driver moved the rest
 * into system RAM (CUDA Sysmem Fallback) - many times slower, and ComfyUI's own out-of-memory fallbacks never start.
 * Say so once per session, with the setting that fixes it. nodes/_c2c_gpu_spill.py measures; this only asks and tells.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const SETTING = "c2c.gpu.spillNotice";
let _enabled = true;
let _shown = false;

async function peak(reset) {
    try {
        const r = await api.fetchApi(`/c2c/gpu/peak?reset=${reset ? 1 : 0}`);
        return r.ok ? await r.json() : null;
    } catch {
        return null;
    }
}

function tell(p) {
    _shown = true;
    const text = `This run asked for ${p.peak_gb} GB of GPU memory on a ${p.total_gb} GB card (${p.device}). `
        + "Windows moved the rest into system RAM, which is many times slower, and ComfyUI's own memory "
        + "fallbacks did not start. Fix: NVIDIA Control Panel › Manage 3D settings › Program settings › add "
        + "ComfyUI's python.exe › CUDA - Sysmem Fallback Policy › Prefer No Sysmem Fallback. "
        + "Measured on an 8 GB laptop: a 2K Wan VAE decode went from 502 s to 37 s.";
    const toast = app.extensionManager?.toast;
    if (toast?.add) {
        toast.add({ severity: "warn", summary: "GPU memory spilled into system RAM", detail: text, life: 30000 });
    } else {
        console.warn(`[C2C] ${text}`);
    }
}

async function onRunEnd() {
    if (!_enabled || _shown) return;
    const p = await peak(true);
    if (p?.available && p.spilled) tell(p);
}

app.registerExtension({
    name: "C2C.GpuSpillNotice",
    settings: [{
        id: SETTING,
        name: "Tell me when GPU memory spills into system RAM",
        tooltip: "After a run, a one-time notice if it used more GPU memory than the card has (Windows' NVIDIA "
            + "driver then uses system RAM, many times slower), with the driver setting that fixes it.",
        type: "boolean",
        defaultValue: true,
        category: ["c2c", "Performance", "GPU memory"],
        onChange: (v) => { _enabled = v !== false; },
    }],
    async setup() {
        _enabled = app.ui.settings.getSettingValue(SETTING) !== false;
        api.addEventListener("execution_start", () => { if (_enabled && !_shown) peak(true); });
        for (const ev of ["execution_success", "execution_error", "execution_interrupted"]) {
            api.addEventListener(ev, onRunEnd);
        }
    },
});
