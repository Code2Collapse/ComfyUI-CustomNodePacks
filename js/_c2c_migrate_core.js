/**
 * Saved-workflow migration for removed / merged C2C nodes - the pure part (no browser imports; tested under Node).
 *
 * The table is core's NodeReplace format (nodes/_legacy_replacements.json): old_node_id, new_node_id,
 * old_widget_ids, input_mapping [{new_id, old_id} | {new_id, set_value}], output_mapping [{old_idx, new_idx}].
 * Semantics follow core 1.52.7 useNodeReplacement.replaceWithMapping (links move with their slot, widget values
 * move by old_widget_ids position, set_value writes a constant, outputs move by index) - applied to the saved JSON
 * BEFORE the graph is configured, so the old type is never "missing". Nothing is dropped silently: a link on an
 * old input or output with no mapping is removed and reported.
 */

const linkArr = (l) => Array.isArray(l);
const L = {
    id: (l) => (linkArr(l) ? l[0] : l.id),
    origin: (l) => (linkArr(l) ? l[1] : l.origin_id),
    originSlot: (l) => (linkArr(l) ? l[2] : l.origin_slot),
    target: (l) => (linkArr(l) ? l[3] : l.target_id),
    targetSlot: (l) => (linkArr(l) ? l[4] : l.target_slot),
    setOriginSlot: (l, v) => { if (linkArr(l)) l[2] = v; else l.origin_slot = v; },
    setTargetSlot: (l, v) => { if (linkArr(l)) l[4] = v; else l.target_slot = v; },
};

/**
 * @param {object} graph - a serialised graph or subgraph definition: { nodes: [...], links: [...] }
 * @param {Map<string, object>} byOld - old_node_id -> replacement row
 * @param {(type: string) => boolean} isRegistered
 * @param {(type: string) => ({inputs: object[], outputs: object[], widgets: string[], values: any[]} | null)} describe
 * @returns {{migrated: object[], problems: string[]}}
 */
export function migrateGraph(graph, byOld, isRegistered, describe) {
    const migrated = [];
    const problems = [];
    if (!graph || !Array.isArray(graph.nodes)) return { migrated, problems };
    const links = Array.isArray(graph.links) ? graph.links : [];
    const linkById = new Map(links.map((l) => [L.id(l), l]));
    const nodeById = new Map(graph.nodes.map((n) => [n.id, n]));
    const dropLink = (id) => {
        const l = linkById.get(id);
        if (!l) return;
        linkById.delete(id);
        const i = links.indexOf(l);
        if (i >= 0) links.splice(i, 1);
        const src = nodeById.get(L.origin(l));
        const out = src?.outputs?.[L.originSlot(l)];
        if (out?.links) out.links = out.links.filter((x) => x !== id);
        const dst = nodeById.get(L.target(l));
        const inp = dst?.inputs?.[L.targetSlot(l)];
        if (inp && inp.link === id) inp.link = null;
    };

    for (const node of graph.nodes) {
        const row = byOld.get(node.type);
        if (!row || isRegistered(node.type)) continue;          // only types that are really gone
        if (!isRegistered(row.new_node_id)) {
            problems.push(`${node.type} (#${node.id}): ${row.new_node_id} is not available, left as it was`);
            continue;
        }
        const info = describe(row.new_node_id);
        if (!info) {
            problems.push(`${node.type} (#${node.id}): could not read ${row.new_node_id}`);
            continue;
        }
        const label = `${node.type} → ${row.new_node_id} (#${node.id})`;
        const oldInputs = Array.isArray(node.inputs) ? node.inputs : [];
        const oldOutputs = Array.isArray(node.outputs) ? node.outputs : [];
        const oldVals = Array.isArray(node.widgets_values) ? node.widgets_values : [];
        const oldWidgetIds = row.old_widget_ids || [];

        const newInputs = info.inputs.map((i) => ({ ...i, link: null }));
        const newOutputs = info.outputs.map((o) => ({ ...o, links: [] }));
        const newVals = info.values.slice();
        const usedOldInputs = new Set();

        for (const m of row.input_mapping || []) {
            if (String(m.new_id).includes(".")) continue;        // Autogrow / DynamicCombo: as core, skipped
            if ("old_id" in m) {
                const oi = oldInputs.findIndex((i) => i.name === m.old_id);
                const ni = newInputs.findIndex((i) => i.name === m.new_id);
                if (oi >= 0 && ni >= 0 && oldInputs[oi].link != null) {
                    const lid = oldInputs[oi].link;
                    newInputs[ni].link = lid;
                    const l = linkById.get(lid);
                    if (l) L.setTargetSlot(l, ni);
                    usedOldInputs.add(oi);
                } else if (oi >= 0) {
                    usedOldInputs.add(oi);
                }
                const wi = oldWidgetIds.indexOf(m.old_id);
                const nw = info.widgets.indexOf(m.new_id);
                if (wi >= 0 && nw >= 0 && oldVals[wi] !== undefined) newVals[nw] = oldVals[wi];
            } else if ("set_value" in m) {
                const nw = info.widgets.indexOf(m.new_id);
                if (nw >= 0) newVals[nw] = m.set_value;
            }
        }
        oldInputs.forEach((inp, oi) => {
            if (inp.link != null && !usedOldInputs.has(oi)) {
                problems.push(`${label}: input "${inp.name}" has no place on ${row.new_node_id}; its link was removed`);
                dropLink(inp.link);
            }
        });

        const movedOut = new Set();
        for (const m of row.output_mapping || []) {
            const src = oldOutputs[m.old_idx];
            if (!src || !newOutputs[m.new_idx]) continue;
            newOutputs[m.new_idx].links = [...(src.links || [])];
            for (const lid of src.links || []) {
                const l = linkById.get(lid);
                if (l) L.setOriginSlot(l, m.new_idx);
            }
            movedOut.add(m.old_idx);
        }
        oldOutputs.forEach((out, oi) => {
            if ((out.links || []).length && !movedOut.has(oi)) {
                problems.push(`${label}: output "${out.name}" has no place on ${row.new_node_id}; its links were removed`);
                for (const lid of [...out.links]) dropLink(lid);
            }
        });

        node.type = row.new_node_id;
        node.inputs = newInputs;
        node.outputs = newOutputs;
        node.widgets_values = newVals;
        if (node.properties && "Node name for S&R" in node.properties) {
            node.properties["Node name for S&R"] = row.new_node_id;
        }
        migrated.push(label);
    }
    return { migrated, problems };
}

/** Root graph and every subgraph definition. */
export function migrateWorkflow(data, table, isRegistered, describe) {
    const byOld = new Map();
    for (const row of table || []) if (row?.old_node_id && !byOld.has(row.old_node_id)) byOld.set(row.old_node_id, row);
    const out = { migrated: [], problems: [] };
    const graphs = [data, ...((data?.definitions?.subgraphs) || [])];
    for (const g of graphs) {
        const r = migrateGraph(g, byOld, isRegistered, describe);
        out.migrated.push(...r.migrated);
        out.problems.push(...r.problems);
    }
    return out;
}
