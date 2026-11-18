// C2C Control Center (L3.04): one panel with what a bug report needs - versions, memory, every loaded C2C pack (its
// version, nodes, id collisions with other packs), C2C failures, settings locked by the environment - and a
// "Copy diagnostics report" button. Data: GET /c2c/control/report (this machine only; secrets already removed on the
// server, nodes/_c2c_control_center.py). Opened from the C2C OmniPill (Tools) or the command palette.
// Nothing runs until the panel opens; no polling.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { mountOmniTool } from "./_c2c_omni_tool.js";

const fmtGB = (b) => (typeof b === "number" ? `${(b / 1024 ** 3).toFixed(1)} GB` : "?");

function el(tag, css, text) {
    const e = document.createElement(tag);
    if (css) e.style.cssText = css;
    if (text != null) e.textContent = String(text);
    return e;
}

function section(title) {
    const box = el("div", "display:flex;flex-direction:column;gap:3px;");
    box.append(el("div", "font:600 11px system-ui;letter-spacing:.06em;text-transform:uppercase;color:var(--c2c-sub);"
        + "margin-top:6px;", title));
    return box;
}

function row(k, v, warn = false) {
    const r = el("div", "display:flex;gap:8px;font:12px system-ui;");
    r.append(el("span", "min-width:150px;color:var(--c2c-sub);", k),
             el("span", `color:${warn ? "var(--c2c-warn, #e5b567)" : "var(--c2c-fg)"};word-break:break-word;`, v ?? "?"));
    return r;
}

/** The report the Copy button puts on the clipboard (pure; the running frontend version is added here). */
export function diagnosticsText(rep, frontendRunning) {
    return JSON.stringify({ ...rep, versions: { ...(rep?.versions || {}), frontend_running: frontendRunning || null } },
                          null, 1);
}

function render(body, rep) {
    body.replaceChildren();
    const fe = window.__COMFYUI_FRONTEND_VERSION__ || null;
    const v = rep.versions || {};
    const s1 = section("Versions");
    s1.append(row("ComfyUI", v.comfyui), row("Frontend running", fe),
              row("Frontend core asks for", v.frontend_required, fe && v.frontend_required && fe !== v.frontend_required),
              row("PyTorch / CUDA", `${v.torch ?? "?"} / ${v.cuda ?? "none"}`), row("Python", v.python), row("System", v.os));
    const m = rep.memory || {};
    const s2 = section("Memory");
    s2.append(row("RAM free / total", `${fmtGB(m.ram_free)} / ${fmtGB(m.ram_total)}`));
    for (const d of m.devices || []) s2.append(row(d.name || d.type, `${fmtGB(d.vram_free)} free of ${fmtGB(d.vram_total)}`));
    if (m.error) s2.append(row("Note", m.error, true));
    const s3 = section("C2C packs");
    for (const p of Array.isArray(rep.packs) ? rep.packs : []) {
        const col = p.collisions?.length
            ? ` - ${p.collisions.length} id(s) taken by another pack: ${p.collisions.map((c) => `${c.id} (${c.won_by})`).join(", ")}`
            : "";
        s3.append(row(p.pack.replace(/^ComfyUI-/, ""), `v${p.version || "?"} - ${p.nodes} nodes${col}`, !!col));
    }
    if (!Array.isArray(rep.packs)) s3.append(row("Packs", rep.packs?.error || "unavailable", true));
    const f = rep.failures || {};
    const s4 = section("Problems");
    const fails = f.failures || [];
    s4.append(row("Failed parts", fails.length ? String(fails.length) : "none", fails.length > 0));
    for (const x of fails.slice(0, 12)) s4.append(row(`${x.group}/${x.key}`, `${x.exception_type}: ${x.message}`, true));
    const s5 = section("Settings set by the environment");
    const locked = (Array.isArray(rep.settings) ? rep.settings : []).filter((x) => x.env_locked);
    s5.append(locked.length ? row("", "") : row("None", "every C2C setting can be changed in Settings"));
    for (const x of locked) s5.append(row(x.id, `${x.env} = ${JSON.stringify(x.value)}`));
    body.append(s1, s2, s3, s4, s5);
}

function build(body) {
    body.style.cssText += "padding:10px 12px;display:flex;flex-direction:column;gap:6px;overflow:auto;";
    const bar = el("div", "display:flex;gap:6px;align-items:center;");
    const content = el("div", "display:flex;flex-direction:column;gap:2px;");
    const note = el("span", "font:11px system-ui;color:var(--c2c-sub);margin-left:auto;");
    let last = null;
    const mkBtn = (label, fn) => {
        const b = el("button", "font:12px system-ui;padding:4px 10px;border-radius:6px;cursor:pointer;"
            + "background:var(--c2c-panelDeep,#1b1930);color:var(--c2c-fg);border:1px solid var(--c2c-border,#3a3557);", label);
        b.type = "button";
        b.addEventListener("click", fn);
        return b;
    };
    const load = async () => {
        note.textContent = "Loading...";
        try {
            const r = await api.fetchApi("/c2c/control/report");
            if (!r.ok) throw new Error(r.status === 403 ? "only on the ComfyUI machine itself" : `HTTP ${r.status}`);
            last = await r.json();
            render(content, last);
            note.textContent = `Updated ${new Date().toLocaleTimeString()}`;
        } catch (e) {
            note.textContent = `Could not load the report: ${e.message}`;
        }
    };
    const copy = async () => {
        if (!last) return;
        const text = diagnosticsText(last, window.__COMFYUI_FRONTEND_VERSION__);
        try {
            await navigator.clipboard.writeText(text);
            note.textContent = "Diagnostics report copied (no keys or tokens in it).";
        } catch (_e) {
            note.textContent = "The browser blocked the clipboard - select the text below instead.";
            const ta = el("textarea", "width:100%;height:140px;font:11px monospace;");
            ta.value = text;
            content.append(ta);
        }
    };
    bar.append(mkBtn("Refresh", load), mkBtn("Copy diagnostics report", copy), note);
    body.append(bar, content);
    return { refresh: load };
}

let refs = null;
let handle = null;
app.registerExtension({
    name: "C2C.ControlCenter",
    commands: [{ id: "c2c.control.open", label: "C2C: Open Control Center", function: () => handle?.open?.() }],
    setup() {
        try {
            handle = mountOmniTool({
                id: "control-center", title: "C2C Control Center", label: "Control", icon: "🩺",
                section: "tools", order: 10, width: 520, height: 560,
                build: (body) => { refs = build(body); return refs; },
                onOpen: () => refs?.refresh?.(),
            });
        } catch (e) {
            console.warn("[C2C] Control Center not mounted:", e);
        }
    },
});
