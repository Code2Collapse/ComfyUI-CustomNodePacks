// C2C Bypass Manager (L4.03 slice 2; design docs/research/BYPASS_MANAGER_design.md in the C2C workspace).
// Bypass / Mute / Normal for every node of a type (through subgraphs), from the node menu, the canvas menu, commands
// and hotkeys. Each action is ONE undo step (app.canvas.emitBeforeChange / emitAfterChange, verified in fe 1.53.10).
// Pinned / locked nodes are left alone. Core keys stay core's: Ctrl+B bypass, Ctrl+M mute (selected nodes).
import { app } from "../../scripts/app.js";
import { MODE, MODE_NAME, walk, match, apply, summary } from "./c2c_bypass/engine.js";

const EXT = "C2C.BypassManager";
const SETTING_NEW = "C2C.BypassManager.ApplyToNew";
const CMD = {
    BYPASS: "c2c.bypass.type.bypass",
    MUTE: "c2c.bypass.type.mute",
    NORMAL: "c2c.bypass.type.normal",
    REGEX: "c2c.bypass.regex",
};

function rootGraph() {
    return app.rootGraph ?? app.graph;
}

function toast(severity, summaryText, detail) {
    try {
        app.extensionManager?.toast?.add({ severity, summary: summaryText, detail, life: 3500 });
    } catch (_e) {
        console.log(`[C2C] ${summaryText}: ${detail}`);
    }
}

/** Run `fn` as one undo step and redraw. */
function oneStep(fn) {
    const c = app.canvas;
    c?.emitBeforeChange?.();
    try {
        return fn();
    } finally {
        c?.emitAfterChange?.();
        c?.setDirty?.(true, true);
        rootGraph()?.setDirtyCanvas?.(true, true);
    }
}

function setTypes(types, mode) {
    const entries = walk(rootGraph());
    const hits = types.flatMap((t) => match(entries, { by: "type", value: t }));
    const r = oneStep(() => apply(hits, mode));
    const what = types.length === 1 ? `every ${types[0]}` : `${types.length} node types`;
    toast("info", `${MODE_NAME[mode][0].toUpperCase()}${MODE_NAME[mode].slice(1)}: ${what}`,
          `${r.changed.length} node(s) changed` + (r.skipped.length ? `, ${r.skipped.length} pinned / locked kept` : ""));
    return r;
}

function selectedTypes() {
    const sel = Object.values(app.canvas?.selected_nodes ?? {});
    return [...new Set(sel.map((n) => n.comfyClass ?? n.type).filter(Boolean))];
}

function setSelectedTypes(mode) {
    const types = selectedTypes();
    if (!types.length) {
        toast("warn", "Bypass Manager", "Select a node first: every node of its type is changed.");
        return;
    }
    setTypes(types, mode);
}

async function setByPattern() {
    const pattern = window.prompt("Bypass Manager - nodes whose type or title match (regular expression):", "");
    if (!pattern) return;
    const what = window.prompt("Set them to: bypass, mute or normal?", "bypass");
    const mode = { bypass: MODE.BYPASS, mute: MODE.MUTE, normal: MODE.NORMAL }[String(what ?? "").trim().toLowerCase()];
    if (mode == null) {
        toast("warn", "Bypass Manager", "Type bypass, mute or normal.");
        return;
    }
    let hits;
    try {
        hits = match(walk(rootGraph()), { by: "regex", value: pattern });
    } catch (e) {
        toast("error", "Bypass Manager", e.message);
        return;
    }
    const r = oneStep(() => apply(hits, mode));
    toast("info", `Bypass Manager: /${pattern}/`, `${hits.length} matched, ${r.changed.length} changed` +
          (r.skipped.length ? `, ${r.skipped.length} pinned / locked kept` : ""));
}

/** Apply-to-new: a node ADDED later (not loaded) takes its type's mode when every other node of that type agrees. */
function adoptTypeMode(node) {
    if (app.configuringGraph) return;                        // loading a workflow: saved modes stay as saved
    queueMicrotask(() => {
        if (!app.ui?.settings?.getSettingValue?.(SETTING_NEW)) return;
        if (!node.graph || (node.mode ?? 0) !== MODE.NORMAL) return;       // a pasted node keeps its copied mode
        const t = node.comfyClass ?? node.type;
        const others = match(walk(rootGraph()), { by: "type", value: t }).filter((e) => e.node !== node);
        const modes = new Set(others.map((e) => e.node.mode ?? MODE.NORMAL));
        if (others.length && modes.size === 1) {
            const m = [...modes][0];
            if (m === MODE.BYPASS || m === MODE.MUTE) {
                node.mode = m;
                node.setDirtyCanvas?.(true, true);
            }
        }
    });
}

app.registerExtension({
    name: EXT,
    settings: [{
        id: SETTING_NEW,
        name: "C2C › Bypass Manager › New nodes follow their type",
        tooltip: "A node you add takes the mode of its type when every other node of that type is bypassed (or "
            + "muted). Loading a workflow never changes saved modes.",
        type: "boolean",
        defaultValue: false,
        category: ["c2c", "Canvas", "Bypass Manager"],
    }],
    nodeCreated(node) {
        adoptTypeMode(node);
    },
    commands: [
        { id: CMD.BYPASS, label: "C2C: Bypass every node of the selected types", function: () => setSelectedTypes(MODE.BYPASS) },
        { id: CMD.MUTE, label: "C2C: Mute every node of the selected types", function: () => setSelectedTypes(MODE.MUTE) },
        { id: CMD.NORMAL, label: "C2C: Set every node of the selected types to normal", function: () => setSelectedTypes(MODE.NORMAL) },
        { id: CMD.REGEX, label: "C2C: Bypass / mute / normal by name pattern", function: setByPattern },
    ],
    keybindings: [
        { combo: { key: "b", ctrl: true, alt: true, shift: true }, commandId: CMD.BYPASS },
        { combo: { key: "m", ctrl: true, alt: true, shift: true }, commandId: CMD.MUTE },
        { combo: { key: "n", ctrl: true, alt: true, shift: true }, commandId: CMD.NORMAL },
    ],
    getNodeMenuItems(node) {
        const t = node?.comfyClass ?? node?.type;
        if (!t) return [];
        const count = match(walk(rootGraph()), { by: "type", value: t }).length;
        return [
            null,
            {
                content: `Every ${t} (${count})`,
                has_submenu: true,
                submenu: {
                    options: [
                        { content: "Bypass all", callback: () => setTypes([t], MODE.BYPASS) },
                        { content: "Mute all", callback: () => setTypes([t], MODE.MUTE) },
                        { content: "Set all to normal", callback: () => setTypes([t], MODE.NORMAL) },
                    ],
                },
            },
        ];
    },
    getCanvasMenuItems() {
        const s = summary(walk(rootGraph())).filter((x) => x.state !== "normal");
        return [
            null,
            { content: "Bypass / mute by name pattern...", callback: setByPattern },
            ...(s.length ? [{
                content: `Restore ${s.reduce((a, x) => a + x.count - x.normal, 0)} bypassed / muted node(s)`,
                callback: () => setTypes(s.map((x) => x.type), MODE.NORMAL),
            }] : []),
        ];
    },
});
