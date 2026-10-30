/**
 * Automatic migration of removed / merged C2C nodes when a workflow loads (D0.14, L7.65 consolidation).
 *
 * Core 1.52.7 only OFFERS a replacement (error panel, a click per type). The owner asked for automatic migration, so
 * before the graph is configured every node whose C2C type is gone is rewritten to its replacement from the same
 * table core uses for API prompts (nodes/_legacy_replacements.json, served at /c2c/legacy_replacements). One notice
 * says what moved and anything that could not. Setting: Settings › C2C › Workflows › Migrate old C2C nodes on load
 * (off = core's own offer).
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { keepWorkflowLinkSlots, migrateWorkflow } from "./_c2c_migrate_core.js";

const SETTING = "c2c.migrate.auto";
// L7.65 P08: survivors whose saved inputs can sit on other slots than the current definition's (Inpaint Stitch Pro
// gained the original_image / mask sockets, which the front end puts before every widget slot): links into them are
// re-slotted by input name before configure (keepLinkSlots in _c2c_migrate_core.js)
// L7.65 P21: Face/Pose Delta Editor (now in ComfyUI-WanAnimatePreprocessV2) gained a pose_data socket, which moves its
// widget slots by one
const KEEP_SLOTS = new Set(["InpaintCropProMEC", "InpaintStitchProMEC", "FacePoseDeltaCoreMEC"]);
let _enabled = true;
let _table = null;

function loadTable() {
    if (!_table) {
        _table = api.fetchApi("/c2c/legacy_replacements")
            .then((r) => (r.ok ? r.json() : []))
            .catch(() => []);
    }
    return _table;
}

const _shapes = new Map();
function describe(type) {
    if (_shapes.has(type)) return _shapes.get(type);
    let info = null;
    try {
        const n = LiteGraph.createNode(type);
        if (n) {
            // configure() hands widgets_values out in order to the widgets that serialize; match that list
            const ws = (n.widgets || []).filter((w) => w.serialize !== false);
            info = {
                inputs: (n.inputs || []).map((i) => {
                    const o = { name: i.name, type: i.type };
                    if (i.widget) o.widget = { name: i.widget.name };
                    if (i.shape != null) o.shape = i.shape;
                    return o;
                }),
                outputs: (n.outputs || []).map((o) => ({ name: o.name, type: o.type })),
                widgets: ws.map((w) => w.name),
                // as LGraphNode.serialize writes them: objects JSON-cloned, undefined -> null
                values: ws.map((w) => (typeof w.value === "object" && w.value
                    ? JSON.parse(JSON.stringify(w.value)) : (w.value ?? null))),
                required: Object.keys(LiteGraph.registered_node_types[type]?.nodeData?.input?.required || {}),
            };
            n.onRemoved?.();
        }
    } catch { /* unknown type */ }
    _shapes.set(type, info);
    return info;
}

const isRegistered = (type) => !!LiteGraph.registered_node_types?.[type];

function tell(report) {
    if (!report.migrated.length && !report.removed.length && !report.problems.length) return;
    const end = (s) => (/[.!?]$/.test(s) ? s : `${s}.`);
    const list = (xs) => end(xs.map((x) => x.replace(/[.]$/, "")).join("; "));
    const lines = [];
    if (report.migrated.length) lines.push(`Updated ${report.migrated.length} old C2C node(s): ${list(report.migrated)}`);
    if (report.removed.length) lines.push(`Removed (no successor): ${list(report.removed)}`);
    if (report.notes.length) lines.push(`Check: ${report.notes.map(end).join(" ")}`);
    if (report.problems.length) lines.push(`Not carried over: ${list(report.problems)}`);
    const attention = report.problems.length || report.notes.length || report.removed.length;
    const toast = app.extensionManager?.toast;
    if (toast?.add) {
        toast.add({ severity: attention ? "warn" : "info", summary: "C2C: old nodes updated",
                    detail: lines.join(" "), life: attention ? 30000 : 8000 });
    }
    console.info(`[C2C] ${lines.join(" ")}`);
}

app.registerExtension({
    name: "C2C.AutoMigrate",
    settings: [{
        id: SETTING,
        name: "Migrate old C2C nodes on load",
        tooltip: "When a workflow uses a C2C node that was merged or removed, replace it with its successor as the "
            + "workflow loads (links and values carried over, anything that cannot be is listed). Off = ComfyUI asks.",
        type: "boolean",
        defaultValue: true,
        category: ["c2c", "Workflows", "Migration"],
        onChange: (v) => { _enabled = v !== false; },
    }],
    async setup() {
        _enabled = app.ui.settings.getSettingValue(SETTING) !== false;
        loadTable();
    },
    async beforeConfigureGraph(graphData) {
        if (!graphData) return;
        if (_enabled) {
            try {
                const table = await loadTable();
                if (table?.length) tell(migrateWorkflow(graphData, table, isRegistered, describe));
            } catch (err) {
                console.warn("[C2C] migration skipped:", err);
            }
        }
        // not a migration: keeps saved links on their inputs, so it runs with the setting off too
        try {
            const types = new Set([...KEEP_SLOTS].filter(isRegistered));
            const moved = keepWorkflowLinkSlots(graphData, types, describe);
            if (moved.length) console.info(`[C2C] links kept on their inputs: ${moved.join("; ")}`);
        } catch (err) {
            console.warn("[C2C] link slot check skipped:", err);
        }
    },
});
