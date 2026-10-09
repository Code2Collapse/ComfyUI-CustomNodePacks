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
import { migrateWorkflow } from "./_c2c_migrate_core.js";

const SETTING = "c2c.migrate.auto";
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
            info = {
                inputs: (n.inputs || []).map((i) => {
                    const o = { name: i.name, type: i.type };
                    if (i.widget) o.widget = { name: i.widget.name };
                    if (i.shape != null) o.shape = i.shape;
                    return o;
                }),
                outputs: (n.outputs || []).map((o) => ({ name: o.name, type: o.type })),
                widgets: (n.widgets || []).map((w) => w.name),
                values: (n.widgets || []).map((w) => (typeof w.value === "object" ? null : w.value)),
            };
            n.onRemoved?.();
        }
    } catch { /* unknown type */ }
    _shapes.set(type, info);
    return info;
}

const isRegistered = (type) => !!LiteGraph.registered_node_types?.[type];

function tell(report) {
    if (!report.migrated.length && !report.problems.length) return;
    const lines = [];
    if (report.migrated.length) lines.push(`Updated ${report.migrated.length} old C2C node(s): ${report.migrated.join("; ")}.`);
    if (report.problems.length) lines.push(`Not carried over: ${report.problems.join("; ")}.`);
    const toast = app.extensionManager?.toast;
    if (toast?.add) {
        toast.add({ severity: report.problems.length ? "warn" : "info", summary: "C2C: old nodes updated",
                    detail: lines.join(" "), life: report.problems.length ? 20000 : 8000 });
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
        if (!_enabled || !graphData) return;
        try {
            const table = await loadTable();
            if (!table?.length) return;
            tell(migrateWorkflow(graphData, table, isRegistered, describe));
        } catch (err) {
            console.warn("[C2C] migration skipped:", err);
        }
    },
});
