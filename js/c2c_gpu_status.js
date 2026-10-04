/**
 * c2c_gpu_status.js — software-GPU detection guidance and status dialog.
 *
 * When the idle WebGL probe (from _c2c_lite.js setup) finds software rendering,
 * shows one warn toast per page load and registers a command that opens a
 * panel with renderer details and browser-specific fix steps.
 */
import { app } from "../../scripts/app.js";
import { buildPanel, bringToFront } from "./_c2c_window.js";
import { getRuntime } from "./_c2c_runtime.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";

const PANEL_ID = "c2c-gpu-status-panel";
const LS_GPU = "c2c.perf.gpu";
const LS_TOAST_SEEN = "c2c.gpu.toastSeen";

let _toastShownThisPage = false;

function _readGpuEntry() {
    try {
        const raw = localStorage.getItem(LS_GPU);
        if (!raw) return null;
        const o = JSON.parse(raw);
        return (o && typeof o === "object") ? o : null;
    } catch (_) {
        return null;
    }
}

function _readToastSeen() {
    try {
        const raw = localStorage.getItem(LS_TOAST_SEEN);
        if (!raw) return [];
        const a = JSON.parse(raw);
        return Array.isArray(a) ? a : [];
    } catch (_) {
        return [];
    }
}

function _markToastSeen(renderer) {
    try {
        const seen = _readToastSeen();
        if (!seen.includes(renderer)) seen.push(renderer);
        localStorage.setItem(LS_TOAST_SEEN, JSON.stringify(seen));
    } catch (_) {}
}

function _rendererMode() {
    try {
        const on = app?.ui?.settings?.getSettingValue?.("Comfy.VueNodes.Enabled") === true;
        return on ? "Nodes 2.0" : "Classic";
    } catch (_) {
        return "Unknown";
    }
}

function _flagsUrl() {
    const ua = String(navigator?.userAgent || "").toLowerCase();
    if (ua.includes("edg/")) return "edge://flags/#ignore-gpu-blocklist";
    if (ua.includes("brave")) return "brave://flags/#ignore-gpu-blocklist";
    return "chrome://flags/#ignore-gpu-blocklist";
}

async function _copyText(text) {
    try {
        if (navigator.clipboard?.writeText) {
            await navigator.clipboard.writeText(text);
            return true;
        }
    } catch (_) {}
    try {
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.left = "-9999px";
        document.body.appendChild(ta);
        ta.select();
        const ok = document.execCommand("copy");
        ta.remove();
        return ok;
    } catch (_) {
        return false;
    }
}

function _fixStepsHtml(vueNodes) {
    const lines = [
        "<p><strong>Chrome / Brave / Edge</strong></p>",
        "<ol>",
        "<li>Settings → System → turn on <em>Use graphics acceleration when available</em>.</li>",
        `<li>Open <code>${_flagsUrl()}</code>, set <em>Override software rendering list</em> to Enabled, relaunch.</li>`,
        "<li>Confirm <code>chrome://gpu</code> shows <em>Canvas: Hardware accelerated</em> and <em>WebGL: Hardware accelerated</em>.</li>",
        "</ol>",
        "<p><strong>Firefox</strong></p>",
        "<ol>",
        "<li><code>about:support</code> → Graphics → Compositing should say <em>WebRender</em> (not WebRender Software).</li>",
        "<li>If software: set <code>gfx.webrender.all</code> = <code>true</code> in <code>about:config</code> and restart.</li>",
        "</ol>",
    ];
    if (vueNodes) {
        lines.push(
            "<p><strong>Nodes 2.0 + software rendering</strong></p>",
            "<p>Nodes 2.0 draws every node as page elements; without the GPU it costs about 3× more (measured). "
            + "The classic renderer is lighter: Settings → Lite Graph → Nodes 2.0 off.</p>",
        );
    }
    return lines.join("\n");
}

function _esc(s) {
    return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

export function openGpuStatusDialog() {
    try {
        const entry = _readGpuEntry() || {};
        const rt = getRuntime();
        const tier = rt?.tier?.() ?? "unknown";
        const perfMode = rt?.readPerfMode?.() ?? "Auto (recommended)";
        const renderer = String(entry.renderer || "(unknown)");
        const software = entry.software === true;
        const vueNodes = _rendererMode() === "Nodes 2.0";
        const flags = _flagsUrl();

        const existing = document.getElementById(PANEL_ID);
        if (existing) {
            bringToFront(existing);
            return;
        }

        const panel = buildPanel({
            id: PANEL_ID,
            title: "GPU & Performance",
            width: 520,
            height: 520,
            storageKey: "gpu-status",
            actions: [{
                id: "copy-flags",
                label: "Copy flags URL",
                title: "Copy browser flags URL to clipboard",
                onClick: async () => {
                    const ok = await _copyText(flags);
                    const st = document.querySelector(`#${PANEL_ID} .status`);
                    if (st) st.textContent = ok ? "Copied flags URL." : "Copy failed — select the URL manually.";
                },
            }],
        });

        const body = panel.body;
        body.innerHTML = `
            <table style="width:100%;border-collapse:collapse;font-size:12px;line-height:1.5">
              <tr><td style="color:var(--c2c-sub);padding:4px 8px 4px 0">Renderer</td><td><code>${_esc(renderer)}</code></td></tr>
              <tr><td style="color:var(--c2c-sub);padding:4px 8px 4px 0">Software rendering</td><td>${software ? "Yes" : "No"}</td></tr>
              <tr><td style="color:var(--c2c-sub);padding:4px 8px 4px 0">C2C tier</td><td>${_esc(tier)}</td></tr>
              <tr><td style="color:var(--c2c-sub);padding:4px 8px 4px 0">Performance mode</td><td>${_esc(perfMode)}</td></tr>
              <tr><td style="color:var(--c2c-sub);padding:4px 8px 4px 0">Renderer mode</td><td>${_esc(_rendererMode())}</td></tr>
            </table>
            <div style="margin-top:12px;font-size:12px;color:var(--c2c-fg)">${_fixStepsHtml(vueNodes)}</div>`;
        if (panel.status) panel.status.textContent = "Advice only — C2C never changes ComfyUI core settings.";
        bringToFront(panel.el);
    } catch (e) {
        __c2cReport("c2c_gpu_status:open", e);
    }
}

/** Called from _c2c_lite.js after idle GPU probe when software === true. */
export function onSoftwareProbeResult(entry) {
    if (!entry || entry.software !== true) return;
    if (_toastShownThisPage) return;
    _toastShownThisPage = true;

    const renderer = String(entry.renderer || "(unknown)");
    _markToastSeen(renderer);

    try {
        const t = app.extensionManager?.toast;
        if (t?.add) {
            t.add({
                severity: "warn",
                summary: "GPU rendering issue",
                detail: `Your browser is drawing ComfyUI without the GPU (${renderer}). `
                       + "C2C switched to Lite. See how to fix — open "
                       + '"C2C: GPU & performance status" from the command palette.',
                life: 12000,
            });
        }
    } catch (e) {
        __c2cReport("c2c_gpu_status:toast", e);
    }
}

if (!(app.extensions || []).some((e) => e?.name === "C2C.GpuStatus")) {
    app.registerExtension({
        name: "C2C.GpuStatus",
        commands: [{
            id: "C2C.gpu.status",
            label: "C2C: GPU & performance status",
            function: () => openGpuStatusDialog(),
        }],
    });
}
