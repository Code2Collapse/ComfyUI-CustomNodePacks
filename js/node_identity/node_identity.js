// C2C Node Identity - extension entry (docs/node-identity.md; design docs/research/node_identity_design.md).
// Global / per-node title visibility + deterministic semantic renaming, controlled by the admin (the ComfyUI page
// on this machine; the server decides every write action: POST /c2c/node_identity/authorize).
// Never changes node type, class, id, inputs, outputs, links or widget values; execution is unchanged.
// License: Apache-2.0
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { propose, resolvePasteCollisions, checkManualAlias, DEFAULT_RULES } from "./engine.js";
import { snapshot, selectedKeys, groupKeys, setAlias, restore, setTitleOverride, identityOf, oneStep, walk }
    from "./adapter.js";
import { createTitleController, overrideOf } from "./titles.js";
import { mountIdentityPanel } from "./panel.js";

const SET_HIDE = "c2c.nodeIdentity.hideTitles";
const SET_HOVER = "c2c.nodeIdentity.hoverReveal";
const CMD = {
    TOGGLE: "c2c.nodeIdentity.toggleTitles", SELECTED: "c2c.nodeIdentity.renameSelected",
    WORKFLOW: "c2c.nodeIdentity.renameWorkflow", GROUP: "c2c.nodeIdentity.renameGroup",
    PREVIEW: "c2c.nodeIdentity.preview", RESTORE: "c2c.nodeIdentity.restore",
    REGENERATE: "c2c.nodeIdentity.regenerate", CONFIGURE: "c2c.nodeIdentity.configure", OPEN: "c2c.nodeIdentity.open",
};

const titles = createTitleController(app);
const server = {
    loaded: false, reachable: false, is_admin: false,
    policy: { ...DEFAULT_RULES, enabled: true, others_may_rename: false, others_may_toggle: true, locked: false,
              ai_naming: false },
    may: { rename: false, toggle: true, configure: false },
};
let namer = null;            // optional AI namer (window.C2CNodeIdentity.registerNamer); used only if the admin allows
let configuring = 0;         // > 0 while a workflow is being loaded (no paste fix-ups then)
let pendingNew = new Set();
let pendingTimer = null;
let panel = null;

const root = () => app.rootGraph ?? app.graph;

function toast(severity, summary, detail) {
    try {
        app.extensionManager?.toast?.add?.({ severity, summary, detail, life: severity === "error" ? 6000 : 3500 });
    } catch (_e) {
        console.log(`[C2C Node Identity] ${summary}: ${detail}`);
    }
}

function settingGet(id, dflt) {
    try {
        const v = app.extensionManager?.setting?.get?.(id);
        return v === undefined || v === null ? dflt : v;
    } catch (_e) {
        return dflt;
    }
}

async function settingSet(id, value) {
    try {
        await app.extensionManager?.setting?.set?.(id, value);
        return true;
    } catch (_e) {
        return false;
    }
}

export function rules() {
    const p = server.policy || {};
    return { template: p.template, separator: p.separator, use_prompt_words: p.use_prompt_words,
             keep_manual: p.keep_manual, excluded_types: p.excluded_types || [] };
}

async function loadState() {
    try {
        const r = await api.fetchApi("/c2c/node_identity/state", { cache: "no-store" });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const s = await r.json();
        Object.assign(server, { loaded: true, reachable: true, is_admin: !!s.is_admin, policy: s.policy, may: s.may });
    } catch (e) {
        // no server answer: fail closed for names (the server decides who may rename), titles stay a local choice
        Object.assign(server, { loaded: true, reachable: false, is_admin: false,
                                may: { rename: false, toggle: true, configure: false } });
        console.warn("[C2C Node Identity] server rules unavailable; renaming is off:", e);
    }
    titles.set({ enabled: server.policy?.enabled !== false });
    panel?.refresh?.();
    return server;
}

/** The server's answer for one write action. */
async function authorize(action) {
    if (!server.reachable) {
        if (action === "toggle") return { allowed: true, reason: "" };
        return { allowed: false, reason: "The Node Identity server route is not available, so renaming is off." };
    }
    try {
        const r = await api.fetchApi("/c2c/node_identity/authorize", {
            method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action }),
        });
        const j = await r.json();
        return { allowed: !!j.allowed, reason: j.reason || j.message || "" };
    } catch (e) {
        return { allowed: false, reason: `Could not reach the server (${e?.message || e}).` };
    }
}

async function allowed(action) {
    const a = await authorize(action);
    if (!a.allowed) toast("warn", "Node Identity", a.reason || "Not allowed.");
    return a.allowed;
}

// ── title visibility ──────────────────────────────────────────────────────────────────────────────────────────
async function toggleTitles() {
    if (server.policy?.enabled === false) {
        toast("info", "Node Identity", "Turned off by the admin; titles stay visible.");
        return;
    }
    if (!(await allowed("toggle"))) return;
    const next = !titles.state.hide;
    titles.set({ hide: next });
    await settingSet(SET_HIDE, next);              // the change event re-applies the same value (idempotent)
    panel?.refresh?.();
}

async function setNodeOverride(node, value) {
    if (!(await allowed("toggle"))) return;
    oneStep(app, () => setTitleOverride(node, value));
    titles.refresh();
}

// ── renaming ──────────────────────────────────────────────────────────────────────────────────────────────────
/** Proposals for a scope: "selected" | "group" | "workflow"; mode "rename" | "regenerate". */
export function preview(scope, mode = "rename") {
    const { snap, refs } = snapshot(root());
    let keys = null;
    let note = "";
    if (scope === "selected") {
        keys = selectedKeys(app, refs);
        if (!keys.size) note = "Select one or more nodes first.";
    } else if (scope === "group") {
        const g = groupKeys(app, refs);
        keys = g.keys;
        if (!g.groups) note = "Select a group (or a node inside one) first.";
        else if (!keys.size) note = "The selected group holds no node.";
    }
    const useNamer = server.policy?.ai_naming && typeof namer === "function" ? namer : null;
    const rows = keys && !keys.size ? [] : propose(snap, { scope: keys, mode, rules: rules(), namer: useNamer });
    const paths = new Map(walk(root()).map((x) => [x.node, x.path]));
    for (const r of rows) {
        const e = refs.get(r.key);
        r.path = e ? (paths.get(e.node) || "") : "";
        r.type = e ? String(e.node.comfyClass || e.node.type || "") : "";
        r.id = e ? e.node.id : r.key;
    }
    return { rows, refs, snap, note };
}

/** Apply rows (status rename) plus manual edits {key: alias}; one undo step. */
export async function applyRows(rows, refs, edits = {}) {
    if (!(await allowed(Object.keys(edits).length ? "edit" : "rename"))) return { changed: 0 };
    let changed = 0;
    const problems = [];
    // the graph may have changed since the preview: never write a name another node now carries
    const autoKeys = new Set(rows.filter((r) => r.status === "rename" && !Object.prototype.hasOwnProperty.call(edits, r.key))
        .map((r) => r.key));
    const keyOfNode = new Map([...refs].map(([k, e]) => [e.node, k]));
    const others = new Set();
    for (const { key, node } of walk(root())) {
        if (!autoKeys.has(keyOfNode.get(node) ?? key)) others.add(String(node.title ?? ""));
    }
    oneStep(app, () => {
        for (const r of rows) {
            const e = refs.get(r.key);
            if (!e) continue;
            if (autoKeys.has(r.key) && others.has(r.proposed)) {
                problems.push(`${r.current}: "${r.proposed}" is used by another node now; preview again.`);
                continue;
            }
            if (Object.prototype.hasOwnProperty.call(edits, r.key)) {
                const { snap } = snapshot(root());
                const chk = checkManualAlias(snap, r.key, edits[r.key], rules());
                if (!chk.ok) { problems.push(`${r.current}: ${chk.reason}`); continue; }
                if (chk.alias !== e.node.title) { setAlias(e.node, chk.alias, "manual"); changed++; }
                continue;
            }
            if (r.status !== "rename" || r.proposed === e.node.title) continue;
            setAlias(e.node, r.proposed, "auto");
            changed++;
        }
    });
    titles.refresh();
    if (problems.length) toast("warn", "Node Identity", problems.join("\n"));
    return { changed, problems };
}

/** Restore original titles of a scope ("selected" | "group" | "workflow"); one undo step. */
export async function restoreScope(scope) {
    if (!(await allowed("restore"))) return 0;
    const { refs } = snapshot(root());
    let keys = null;
    if (scope === "selected") keys = selectedKeys(app, refs);
    else if (scope === "group") keys = groupKeys(app, refs).keys;
    let n = 0;
    oneStep(app, () => {
        for (const [k, e] of refs) {
            if (keys && !keys.has(k)) continue;
            if (restore(e.node)) n++;
        }
    });
    titles.refresh();
    toast("info", "Node Identity", n ? `${n} title(s) restored.` : "Nothing to restore here.");
    return n;
}

async function renameOne(node) {
    const cur = String(node.title ?? "");
    let value = null;
    try {
        value = await app.extensionManager?.dialog?.prompt?.({ title: "Node Identity", message: "Name for this node:",
                                                               defaultValue: cur });
    } catch (_e) {
        value = window.prompt("Name for this node:", cur);
    }
    if (value == null) return;
    const { refs } = snapshot(root());
    const key = [...refs].find(([, e]) => e.node === node)?.[0];
    if (key == null) return;
    await applyRows([{ key, current: cur, proposed: value, status: "edit" }], refs, { [key]: value });
}

// ── paste / duplicate: only freshly created nodes, never during a workflow load ───────────────────────────────
function onNodeCreated(node) {
    if (configuring > 0) return;
    pendingNew.add(node);
    if (pendingTimer) return;
    pendingTimer = setTimeout(() => {
        pendingTimer = null;
        const fresh = pendingNew;
        pendingNew = new Set();
        if (configuring > 0 || server.policy?.enabled === false) { titles.refresh(); return; }
        const { snap, refs } = snapshot(root());
        const keys = [...refs].filter(([, e]) => fresh.has(e.node) && identityOf(e.node)).map(([k]) => k);
        const fix = keys.length ? resolvePasteCollisions(snap, keys, rules()).filter((r) => r.status === "rename") : [];
        if (fix.length && !server.may?.rename) {
            toast("warn", "Node Identity", `${fix.length} pasted node(s) repeat an existing name; names are locked `
                + "for this client, so they were left as they are.");
        } else if (fix.length) {
            // our own alias on a copy: keep names unique (no extra round-trip: the server's answer is in `server.may`;
            // it only re-numbers an alias the copy already carried)
            oneStep(app, () => { for (const r of fix) setAlias(refs.get(r.key).node, r.proposed, "auto"); });
        }
        titles.refresh();
    }, 0);
}

// ── menus ─────────────────────────────────────────────────────────────────────────────────────────────────────
const run = (id) => app.extensionManager?.command?.execute?.(id);

function canvasMenu() {
    if (server.policy?.enabled === false) return [];
    return [{
        content: "Node Identity",
        submenu: { options: [
            { content: titles.state.hide ? "Show titles" : "Hide titles", callback: () => run(CMD.TOGGLE) },
            { content: "Smart rename selected", callback: () => run(CMD.SELECTED) },
            { content: "Rename selected group", callback: () => run(CMD.GROUP) },
            { content: "Rename entire workflow", callback: () => run(CMD.WORKFLOW) },
            { content: "Preview proposed names…", callback: () => run(CMD.PREVIEW) },
            { content: "Restore original titles", callback: () => run(CMD.RESTORE) },
        ] },
    }];
}

function nodeMenu(node) {
    if (server.policy?.enabled === false || !node) return [];
    const o = overrideOf(node);
    const mark = (v) => (o === v ? "● " : "○ ");
    return [{
        content: "Node Identity",
        submenu: { options: [
            { content: `${mark(null)}Title: follow global`, callback: () => setNodeOverride(node, null) },
            { content: `${mark("show")}Title: always show`, callback: () => setNodeOverride(node, "show") },
            { content: `${mark("hide")}Title: always hide`, callback: () => setNodeOverride(node, "hide") },
            { content: "Rename this node…", callback: () => renameOne(node) },
            { content: "Smart rename selected", callback: () => run(CMD.SELECTED) },
            { content: "Restore original title", callback: async () => {
                if (!(await allowed("restore"))) return;
                oneStep(app, () => restore(node));
                titles.refresh();
            } },
        ] },
    }];
}

function openPanel(view) {
    panel?.open?.(view);
}

// Public hook: AI namer (off unless the admin turns ai_naming on; deterministic rules stay the fallback) and the
// same actions as the panel for automation (every write still asks the server).
window.C2CNodeIdentity = Object.freeze({
    registerNamer(fn) { namer = typeof fn === "function" ? fn : null; },
    preview: (scope = "workflow", mode = "rename") => preview(scope, mode).rows,
    /** Preview + apply every proposed name of a scope (same permission check as the panel's Apply). */
    async rename(scope = "workflow", mode = "rename") {
        const { rows, refs } = preview(scope, mode);
        return applyRows(rows.filter((r) => r.status === "rename"), refs);
    },
    restore: (scope = "workflow") => restoreScope(scope),
    reloadState: () => loadState(),
    state: () => ({ server: JSON.parse(JSON.stringify(server)), titles: { ...titles.state } }),
});

app.registerExtension({
    name: "C2C.NodeIdentity",
    commands: [
        { id: CMD.TOGGLE, label: "C2C: Toggle Title Visibility", icon: "pi pi-eye-slash", function: toggleTitles },
        { id: CMD.SELECTED, label: "C2C: Smart Rename Selected", icon: "pi pi-tag", function: () => openPanel({ scope: "selected" }) },
        { id: CMD.WORKFLOW, label: "C2C: Smart Rename Workflow", function: () => openPanel({ scope: "workflow" }) },
        { id: CMD.GROUP, label: "C2C: Smart Rename Selected Group", function: () => openPanel({ scope: "group" }) },
        { id: CMD.PREVIEW, label: "C2C: Preview Proposed Names", function: () => openPanel({ scope: "workflow" }) },
        { id: CMD.RESTORE, label: "C2C: Restore Original Titles", function: () => openPanel({ scope: "workflow", restore: true }) },
        { id: CMD.REGENERATE, label: "C2C: Regenerate Node Names", function: () => openPanel({ scope: "workflow", mode: "regenerate" }) },
        { id: CMD.CONFIGURE, label: "C2C: Configure Naming Rules", function: () => openPanel({ admin: true }) },
        { id: CMD.OPEN, label: "C2C: Open Node Identity", function: () => openPanel({}) },
    ],
    keybindings: [{ combo: { key: "t", alt: true, shift: true }, commandId: CMD.TOGGLE }],
    getCanvasMenuItems() {
        return canvasMenu();
    },
    getNodeMenuItems(node) {
        return nodeMenu(node);
    },
    getSelectionToolboxCommands() {
        return server.policy?.enabled === false ? [] : [CMD.SELECTED];
    },
    nodeCreated(node) {
        onNodeCreated(node);
    },
    async beforeConfigureGraph() {
        configuring++;
    },
    async afterConfigureGraph() {
        configuring = Math.max(0, configuring - 1);
        titles.refresh();
    },
    async setup() {
        titles.set({ hide: !!settingGet(SET_HIDE, false), hover: settingGet(SET_HOVER, true) !== false });
        try {
            app.ui?.settings?.addEventListener?.(`${SET_HIDE}.change`, (e) => {
                titles.set({ hide: !!e?.detail?.value });
                panel?.refresh?.();
            });
            app.ui?.settings?.addEventListener?.(`${SET_HOVER}.change`, (e) => titles.set({ hover: e?.detail?.value !== false }));
        } catch (_e) { /* settings events unavailable: the commands still apply the state */ }
        try {   // the graph on screen changed (entered / left a subgraph): Vue rules follow it
            app.canvas?.canvas?.addEventListener?.("litegraph:set-graph", () => setTimeout(() => titles.refresh(), 0));
        } catch (_e) { /* no canvas */ }
        try {
            panel = mountIdentityPanel({
                app, server, titles, preview, applyRows, restoreScope, toggleTitles, loadState,
                savePolicy: async (p) => {
                    const r = await api.fetchApi("/c2c/node_identity/policy", {
                        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(p),
                    });
                    const j = await r.json().catch(() => ({}));
                    if (!r.ok) throw new Error(j.message || `HTTP ${r.status}`);
                    await loadState();
                    return j;
                },
            });
        } catch (e) {
            console.warn("[C2C] Node Identity panel not mounted:", e);
        }
        await loadState();
    },
});
