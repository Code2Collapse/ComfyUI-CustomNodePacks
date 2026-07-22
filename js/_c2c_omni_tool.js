/**
 * _c2c_omni_tool.js — shared helper to surface a C2C tool through OmniPill
 * instead of a sidebar tab.
 *
 * A tool that used to call `registerSidebarTab({ render: el => buildView(el) })`
 * now calls:
 *
 *   mountOmniTool({
 *     id: "prompt-wizard", title: "C2C Prompt Wizard", label: "Prompt", icon: "✍️",
 *     section: "ai", order: 20, width: 460, height: 560,
 *     build: (body, refs) => buildView(body),
 *   });
 *
 * It:
 *   1. registers a chip in the given OmniPill section (C2COmniBar.register),
 *   2. opens the tool as a draggable/resizable floating panel (buildPanel) on click,
 *   3. registers a command-palette entry (Ctrl+K) as a fallback so the tool is
 *      still reachable if OmniPill ever fails to load,
 *   4. does NOT register a sidebar tab — the whole point is to free rail space.
 *
 * The tool's own view builder is reused verbatim; only the mount changes.
 */
import { app } from "../../scripts/app.js";
import { buildPanel, bringToFront } from "./_c2c_window.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";

const CHIP_CSS =
    "background:var(--c2c-bg2);color:var(--c2c-fg);border:1px solid var(--c2c-border);" +
    "border-radius:5px;padding:3px 9px;cursor:pointer;font-size:11px;white-space:nowrap;" +
    "display:inline-flex;align-items:center;gap:5px;line-height:1.6";

export function mountOmniTool({
    id,
    title,
    label,
    icon = "🔧",
    section = "tools",
    order = 100,
    shortcut = null,
    width = 460,
    height = 520,
    build,
    onOpen = null,
}) {
    const panelId = `c2c-omni-tool-${id}`;

    function open() {
        const existing = document.getElementById(panelId);
        if (existing) { bringToFront(existing); return existing; }
        const refs = buildPanel({ id: panelId, title, shortcut, width, height,
                                  storageKey: `omni.tool.${id}` });
        try {
            build(refs.body, refs);
        } catch (err) {
            __c2cReport(`omniTool.build:${id}`, err);
            refs.body.innerHTML =
                `<div style="padding:16px;color:var(--c2c-red,#f38ba8);font:13px system-ui">` +
                `This tool failed to open.<br><span style="opacity:.7">${(err && err.message) || err}</span></div>`;
        }
        bringToFront(refs.el);
        if (typeof onOpen === "function") { try { onOpen(refs); } catch (e) { /* noop */ } }
        return refs.el;
    }

    // ── OmniPill chip ────────────────────────────────────────────────────────
    const chip = document.createElement("button");
    chip.className = "c2c-omni-tool-chip";
    chip.style.cssText = CHIP_CSS;
    const full = `${icon} ${label}`;
    chip.textContent = full;
    chip.title = title;
    chip.addEventListener("pointerenter", () => { chip.style.borderColor = "var(--c2c-blue,#89b4fa)"; });
    chip.addEventListener("pointerleave", () => { chip.style.borderColor = "var(--c2c-border)"; });
    chip.onclick = open;

    let _unregister = () => {};
    const doRegister = (tries = 0) => {
        const api = window.C2COmniBar;
        if (api && typeof api.register === "function") {
            _unregister = api.register({
                section, id, order, element: chip,
                onMode: (m) => { chip.textContent = m === "icon" ? icon : full; },
            }) || (() => {});
            return;
        }
        if (tries < 20) setTimeout(() => doRegister(tries + 1), 150);  // OmniPill still loading
    };
    doRegister();

    // ── Command-palette fallback (always reachable) ──────────────────────────
    try {
        app.extensionManager?.registerCommand?.({
            id: `c2c.omniTool.${id}.open`,
            label: `C2C: Open ${title}`,
            function: open,
        });
    } catch (_) { /* command API may be unavailable on old cores */ }

    return { open, chip, unregister: () => _unregister() };
}
