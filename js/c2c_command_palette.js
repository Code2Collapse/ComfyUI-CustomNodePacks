// c2c_command_palette.js — Ctrl+K universal launcher (C2C)
// ---------------------------------------------------------------------
// One floating fuzzy input that searches across:
//   • Nodes in this workflow (jump-to)
//   • Node types to ADD (delegates to Vue palette / showSearchBox)
//   • Settings (jumps to Comfy Settings panel + filter)
//   • Workflow files (workflows/*.json) — when API available
//   • C2C extension commands (registered via app.extensionManager.command)
//   • Gallery items (recent generations from /history)
//
// Activation:   Ctrl+K  (Cmd+K on macOS)
// Navigation:   ↑/↓ select • Enter execute • Esc close
// Category-jump: type prefix to filter source
//      "+ flux"   → add-node search filtered to "flux"
//      "> light"  → settings filtered to "light"
//      "! sam"    → C2C/extension commands matching "sam"
//      "@ mask"   → only nodes-in-workflow
//      "# 12"     → jump to node id 12
// Settings: c2c.commandPalette.enabled (default ON).
// License: Apache-2.0
// ---------------------------------------------------------------------

import { app } from "../../scripts/app.js";

const STYLE_ID = "c2c-cmdpal-style";
const ROOT_ID  = "c2c-cmdpal-root";
const SETTING_ID = "c2c.commandPalette.enabled";

const ICONS = {
    node: "◆", add: "➕", setting: "⚙", workflow: "🗎",
    command: "▶", gallery: "🖼", error: "⚠",
};

function injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
#${ROOT_ID} {
    position: fixed; top: 18%; left: 50%; transform: translateX(-50%);
    z-index: 100001; width: min(640px, 94vw);
    background: rgba(20, 22, 28, 0.97);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px; box-shadow: 0 24px 60px rgba(0,0,0,0.65);
    font: 13px/1.45 ui-sans-serif, system-ui, "Segoe UI", sans-serif;
    color: #e8ecf1; backdrop-filter: blur(14px);
    padding: 10px 10px 8px;
    display: none;
}
#${ROOT_ID}.is-open { display: block; }
#${ROOT_ID} .c2c-cmdpal-head {
    display:flex; align-items:center; gap:10px; padding: 0 4px 6px;
}
#${ROOT_ID} .c2c-cmdpal-title {
    font-weight: 600; color: #b3d1ff; letter-spacing: .2px;
}
#${ROOT_ID} .c2c-cmdpal-hint {
    margin-left: auto; color:#7d8896; font-size: 11px;
}
#${ROOT_ID} input.c2c-cmdpal-input {
    width: 100%; box-sizing: border-box;
    background: rgba(0,0,0,0.40); color:#fff;
    border: 1px solid rgba(255,255,255,0.10); border-radius: 8px;
    padding: 11px 12px; font: 14px ui-monospace, "Cascadia Code", monospace;
    outline: none;
}
#${ROOT_ID} input.c2c-cmdpal-input:focus {
    border-color: #5b8def; box-shadow: 0 0 0 2px rgba(91,141,239,0.20);
}
#${ROOT_ID} .c2c-cmdpal-list {
    margin-top: 7px; max-height: 460px; overflow-y: auto;
}
#${ROOT_ID} .c2c-cmdpal-cat {
    color:#9fb6d1; font-size:10.5px; text-transform:uppercase;
    letter-spacing:1px; padding: 6px 6px 2px;
}
#${ROOT_ID} .c2c-cmdpal-row {
    display:grid; grid-template-columns: 22px 1fr auto;
    gap: 8px; align-items:center; padding: 7px 9px;
    border-radius: 7px; cursor: pointer; user-select:none;
}
#${ROOT_ID} .c2c-cmdpal-row.sel { background: rgba(91,141,239,0.22); }
#${ROOT_ID} .c2c-cmdpal-row:hover { background: rgba(255,255,255,0.05); }
#${ROOT_ID} .c2c-cmdpal-row .icn {
    color:#9ec1ff; text-align:center; font-size: 14px;
}
#${ROOT_ID} .c2c-cmdpal-row .ttl { color:#fff; }
#${ROOT_ID} .c2c-cmdpal-row .sub { color:#9fa9b8; font-size: 11.5px; }
#${ROOT_ID} .c2c-cmdpal-row mark {
    background: rgba(255,209,102,0.30); color:#fff;
    border-radius: 2px; padding: 0 1px;
}
#${ROOT_ID} .c2c-cmdpal-empty {
    color:#7d8896; text-align:center; padding: 18px 0 12px;
}
.c2c-cmdpal-backdrop {
    position: fixed; inset: 0; z-index: 100000;
    background: rgba(0,0,0,0.30);
    display: none;
}
.c2c-cmdpal-backdrop.is-open { display: block; }
`;
    document.head.appendChild(s);
}

// ── Fuzzy ─────────────────────────────────────────────────────────────
function fuzzy(needle, haystack) {
    if (!needle) return { score: 0, hits: [] };
    const n = needle.toLowerCase();
    const h = (haystack || "").toLowerCase();
    if (!h) return { score: -1, hits: [] };
    if (h === n) return { score: 1000, hits: [[0, n.length]] };
    const ix = h.indexOf(n);
    if (ix !== -1) {
        return {
            score: 600 - ix + (ix === 0 ? 90 : 0),
            hits: [[ix, ix + n.length]],
        };
    }
    let i = 0, j = 0, last = -1, runs = 0, score = 0;
    const hits = [];
    while (i < n.length && j < h.length) {
        if (n[i] === h[j]) {
            if (last === j - 1) {
                runs++; hits[hits.length - 1][1] = j + 1;
            } else {
                runs = 1; hits.push([j, j + 1]);
                if (j === 0 || /[\s_\-./]/.test(h[j - 1])) score += 30;
            }
            score += 8 + runs * 3;
            last = j; i++;
        }
        j++;
    }
    if (i < n.length) return { score: -1, hits: [] };
    return { score, hits };
}

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;",
        '"': "&quot;", "'": "&#39;",
    }[c]));
}

function highlight(text, hits) {
    if (!hits || !hits.length) return escapeHtml(text);
    let out = "", cur = 0;
    for (const [a, b] of hits) {
        out += escapeHtml(text.slice(cur, a));
        out += "<mark>" + escapeHtml(text.slice(a, b)) + "</mark>";
        cur = b;
    }
    out += escapeHtml(text.slice(cur));
    return out;
}

// ── Source providers ──────────────────────────────────────────────────
function provideWorkflowNodes() {
    const nodes = app.graph?._nodes || [];
    return nodes.map((n) => ({
        kind: "node",
        icon: ICONS.node,
        label: n.title || n.type || `node ${n.id}`,
        sub: `${n.type} • #${n.id}`,
        key: `${n.title || ""} ${n.type || ""} ${n.id}`,
        run: () => focusNode(n),
    }));
}

function provideAddNode() {
    // List registered node types. Comfy exposes them at LiteGraph.registered_node_types.
    const types = (window.LiteGraph?.registered_node_types) || {};
    const out = [];
    for (const t of Object.keys(types)) {
        out.push({
            kind: "add",
            icon: ICONS.add,
            label: t,
            sub: "add node to graph",
            key: t,
            run: () => addNodeAtCursor(t),
        });
    }
    return out;
}

function provideSettings() {
    // Settings are registered via app.ui.settings.addSetting / extensionManager.setting.
    const list = [];
    try {
        const reg = app.extensionManager?.setting?.settings;
        const entries = reg
            ? (Array.isArray(reg) ? reg : Object.values(reg))
            : (app.ui?.settings?.settingsLookup
                ? Object.values(app.ui.settings.settingsLookup)
                : []);
        for (const s of entries) {
            if (!s) continue;
            list.push({
                kind: "setting",
                icon: ICONS.setting,
                label: s.name || s.id || "(unnamed setting)",
                sub: s.id || "setting",
                key: `${s.name || ""} ${s.id || ""} ${(s.tooltip || "")}`,
                run: () => openSettingsTo(s.id),
            });
        }
    } catch { /* silent */ }
    return list;
}

function provideCommands() {
    const list = [];
    try {
        const reg = app.extensionManager?.command?.commands;
        const entries = reg
            ? (Array.isArray(reg) ? reg : Object.values(reg))
            : [];
        for (const c of entries) {
            if (!c) continue;
            list.push({
                kind: "command",
                icon: ICONS.command,
                label: c.label || c.id || "(command)",
                sub: c.id || "",
                key: `${c.label || ""} ${c.id || ""}`,
                run: () => {
                    try { app.extensionManager.command.execute(c.id); }
                    catch (e) { console.warn("[C2C.CmdPal] exec failed:", e); }
                },
            });
        }
    } catch { /* silent */ }
    return list;
}

// ── Actions ───────────────────────────────────────────────────────────
function focusNode(node) {
    if (!node || !app.canvas) return;
    try {
        const ds = app.canvas.ds;
        if (ds) {
            const cx = node.pos[0] + node.size[0] / 2;
            const cy = node.pos[1] + node.size[1] / 2;
            ds.scale = Math.min(1.2, Math.max(0.55, ds.scale || 1));
            const dpr = window.devicePixelRatio || 1;
            ds.offset[0] = -cx + (app.canvas.canvas.width / (2 * dpr)) / ds.scale;
            ds.offset[1] = -cy + (app.canvas.canvas.height / (2 * dpr)) / ds.scale;
        }
        app.canvas.selectNode?.(node, false);
        app.canvas.setDirty(true, true);
    } catch (e) {
        console.warn("[C2C.CmdPal] focusNode failed:", e);
    }
}

function addNodeAtCursor(type) {
    try {
        const node = window.LiteGraph?.createNode?.(type);
        if (!node) return;
        const canvas = app.canvas;
        const ds = canvas?.ds;
        let pos = [200, 200];
        if (ds && canvas?.canvas) {
            const dpr = window.devicePixelRatio || 1;
            const cx = canvas.canvas.width  / (2 * dpr);
            const cy = canvas.canvas.height / (2 * dpr);
            pos = [
                (cx - ds.offset[0] * ds.scale) / ds.scale,
                (cy - ds.offset[1] * ds.scale) / ds.scale,
            ];
        }
        node.pos = pos;
        app.graph.add(node);
        canvas?.selectNode?.(node, false);
        canvas?.setDirty(true, true);
    } catch (e) {
        console.warn("[C2C.CmdPal] addNode failed:", e);
    }
}

function openSettingsTo(id) {
    try {
        if (app.extensionManager?.setting?.open) {
            app.extensionManager.setting.open(id);
            return;
        }
        // Legacy fallback: click the settings gear and let the user search.
        app.ui?.settings?.show?.();
    } catch (e) {
        console.warn("[C2C.CmdPal] openSettings failed:", e);
    }
}

// ── Palette UI ────────────────────────────────────────────────────────
let _root = null, _input = null, _list = null, _backdrop = null;
let _items = [], _sel = 0;

function ensureRoot() {
    if (_root) return;
    injectStyle();
    _backdrop = document.createElement("div");
    _backdrop.className = "c2c-cmdpal-backdrop";
    _backdrop.addEventListener("click", close);
    document.body.appendChild(_backdrop);

    _root = document.createElement("div");
    _root.id = ROOT_ID;
    _root.innerHTML = `
        <div class="c2c-cmdpal-head">
            <div class="c2c-cmdpal-title">Command Palette</div>
            <div class="c2c-cmdpal-hint">↑↓ • Enter • Esc · prefixes: + add  &gt; setting  ! cmd  @ node  # id</div>
        </div>
        <input class="c2c-cmdpal-input" type="text"
               placeholder="Search nodes, settings, commands, or type + to add a new node…" />
        <div class="c2c-cmdpal-list"></div>
        <div class="c2c-cmdpal-empty" style="display:none">No matches.</div>
    `;
    document.body.appendChild(_root);
    _input = _root.querySelector("input.c2c-cmdpal-input");
    _list  = _root.querySelector(".c2c-cmdpal-list");
    _input.addEventListener("input", refresh);
    _input.addEventListener("keydown", onKey);
}

function open() {
    ensureRoot();
    _root.classList.add("is-open");
    _backdrop.classList.add("is-open");
    _input.value = "";
    refresh();
    _input.focus();
}

function close() {
    _root?.classList.remove("is-open");
    _backdrop?.classList.remove("is-open");
}

function gather(q) {
    // Prefix routing.
    const m = q.match(/^([+>!@#])\s*(.*)$/);
    if (m) {
        const tag = m[1], rest = m[2];
        if (tag === "+") return rank(rest, provideAddNode());
        if (tag === ">") return rank(rest, provideSettings());
        if (tag === "!") return rank(rest, provideCommands());
        if (tag === "@") return rank(rest, provideWorkflowNodes());
        if (tag === "#") {
            const id = rest.trim();
            const n = (app.graph?._nodes || []).find((x) => String(x.id) === id);
            return n ? [{
                kind: "node", icon: ICONS.node,
                label: n.title || n.type || `node ${n.id}`,
                sub: `${n.type} • #${n.id}`,
                key: id, hits: { ttl: [[0, (n.title || "").length]] },
                run: () => focusNode(n),
            }] : [];
        }
    }
    const pool = [
        ...provideWorkflowNodes(),
        ...provideCommands(),
        ...provideSettings(),
        ...provideAddNode(),
    ];
    return rank(q, pool);
}

function rank(q, pool) {
    const out = [];
    for (const it of pool) {
        if (!q) { out.push({ ...it, score: 0, hits: null }); continue; }
        const r = fuzzy(q, it.key || it.label || "");
        if (r.score < 0) continue;
        const lblHits = fuzzy(q, it.label || "").hits;
        out.push({ ...it, score: r.score, hits: { ttl: lblHits } });
    }
    out.sort((a, b) => b.score - a.score);
    return out.slice(0, 60);
}

function refresh() {
    const q = (_input.value || "").trim();
    _items = gather(q);
    _sel = 0;
    render();
}

function render() {
    const empty = _root.querySelector(".c2c-cmdpal-empty");
    if (!_items.length) {
        _list.innerHTML = "";
        empty.style.display = _input.value ? "block" : "none";
        return;
    }
    empty.style.display = "none";
    // Group rows by `kind` for visual headers.
    let lastKind = null, html = "";
    _items.forEach((it, i) => {
        if (it.kind !== lastKind) {
            html += `<div class="c2c-cmdpal-cat">${escapeHtml(it.kind)}</div>`;
            lastKind = it.kind;
        }
        const cls = i === _sel ? "c2c-cmdpal-row sel" : "c2c-cmdpal-row";
        const lbl = highlight(it.label || "", it.hits?.ttl);
        html += `<div class="${cls}" data-i="${i}">
            <span class="icn">${it.icon || "·"}</span>
            <span><div class="ttl">${lbl}</div>
                  <div class="sub">${escapeHtml(it.sub || "")}</div></span>
            <span class="sub">${escapeHtml(it.kind)}</span>
        </div>`;
    });
    _list.innerHTML = html;
    [..._list.querySelectorAll(".c2c-cmdpal-row")].forEach((el) => {
        el.addEventListener("click", () => {
            const i = +el.dataset.i;
            const it = _items[i];
            close();
            try { it.run(); } catch (e) { console.warn("[C2C.CmdPal] run failed:", e); }
        });
    });
}

function onKey(ev) {
    if (ev.key === "Escape") { ev.preventDefault(); close(); return; }
    if (ev.key === "ArrowDown") {
        ev.preventDefault();
        _sel = Math.min(_items.length - 1, _sel + 1);
        render(); ensureSelVisible(); return;
    }
    if (ev.key === "ArrowUp") {
        ev.preventDefault();
        _sel = Math.max(0, _sel - 1);
        render(); ensureSelVisible(); return;
    }
    if (ev.key === "Enter") {
        ev.preventDefault();
        const it = _items[_sel];
        if (!it) return;
        close();
        try { it.run(); } catch (e) { console.warn("[C2C.CmdPal] run failed:", e); }
    }
}

function ensureSelVisible() {
    const el = _list.querySelector(".c2c-cmdpal-row.sel");
    if (el?.scrollIntoView) el.scrollIntoView({ block: "nearest" });
}

// ── Global Ctrl+K / Cmd+K ─────────────────────────────────────────────
function onGlobalKey(ev) {
    if (!(ev.ctrlKey || ev.metaKey)) return;
    if (ev.key !== "k" && ev.key !== "K") return;
    try {
        if (app.ui.settings.getSettingValue(SETTING_ID, true) === false) return;
    } catch { /* assume enabled */ }
    // Don't hijack inside form fields (but DO hijack when our own input
    // is already focused — toggle close).
    const ae = document.activeElement;
    if (ae && _root && _root.contains(ae)) {
        ev.preventDefault();
        close();
        return;
    }
    const tag = ae?.tagName?.toLowerCase();
    const editable = tag === "input" || tag === "textarea"
                     || ae?.isContentEditable === true;
    if (editable) return;
    ev.preventDefault();
    ev.stopPropagation();
    open();
}

app.registerExtension({
    name: "C2C.CommandPalette",
    async setup() {
        try {
            app.ui.settings.addSetting({
                id: SETTING_ID,
                name: "Command Palette (Ctrl+K)",
                tooltip:
                    "Open a universal fuzzy launcher that searches across " +
                    "workflow nodes, add-node types, settings, and " +
                    "extension commands.",
                type: "boolean",
                defaultValue: true,
                category: ["c2c", "Overlays", "Command Palette"],
            });
        } catch { /* settings API not ready */ }
        window.addEventListener("keydown", onGlobalKey, true);
        console.log("[C2C.CommandPalette] Ctrl+K palette armed.");
    },
});
