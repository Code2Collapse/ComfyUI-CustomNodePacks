/**
 * Saved-workflow migration for removed / merged C2C nodes - the pure part (no browser imports; tested under Node).
 *
 * The table is core's NodeReplace format (nodes/_legacy_replacements.json): old_node_id, new_node_id,
 * old_widget_ids, input_mapping [{new_id, old_id} | {new_id, set_value}], output_mapping [{old_idx, new_idx}].
 * Semantics follow core 1.52.7 useNodeReplacement.replaceWithMapping (links move with their slot, widget values
 * move by old_widget_ids position, set_value writes a constant, outputs move by index) - applied to the saved JSON
 * BEFORE the graph is configured, so the old type is never "missing". Nothing is dropped silently: a link on an
 * old input or output with no mapping is removed and reported.
 *
 * C2C-only row fields (core never sees them; nodes/_legacy_replacements.register passes only core's fields):
 *   c2c_values  [{new_id, old_id, map?, fn?: "log2", clamp?: [min, max]} | {new_id, set}] - a value that needs
 *               translating (another combo vocabulary, a multiplier that became stops); applied after the core mapping.
 *               map keys are String(old value): booleans are "true" / "false"
 *   c2c_note    what behaves differently on the successor - reported once per type
 *   c2c_remove  the node has no successor: it is taken out of the graph, its links removed, all of it reported
 *   c2c_by_value {widget, cases: {<String(saved value)>: {...row fields}}} - a removed multi-mode node whose modes
 *               went to different successors (L7.65 P01): the saved value of that old widget picks the case, whose
 *               fields replace the row's (new_node_id, input_mapping, output_mapping, c2c_values, c2c_note, ...).
 *               No matching case (or no saved value) = the row itself, which is also what core gets registered.
 * A seed's control_after_generate travels with the seed; when the old node had none it becomes "fixed", because the
 * old node never changed its seed by itself.
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
const CONTROL = "control_after_generate";

function translate(v, ov) {
    let nv = ov;
    if (v.map) {
        if (!Object.prototype.hasOwnProperty.call(v.map, String(ov))) return { ok: false };
        nv = v.map[String(ov)];
    }
    if (v.fn === "log2") {
        const x = Number(nv);
        if (!(x > 0)) return { ok: false };
        nv = Math.log2(x);
    }
    if (Array.isArray(v.clamp) && typeof nv === "number") nv = Math.min(v.clamp[1], Math.max(v.clamp[0], nv));
    return { ok: true, value: nv };
}

/**
 * The row that applies to one saved node: a c2c_by_value case picked by the node's saved value, else the row.
 * @param {object} row - replacement row
 * @param {any[]} widgetsValues - the old node's saved widgets_values
 * @returns {object}
 */
export function pickCase(row, widgetsValues) {
    const by = row?.c2c_by_value;
    if (!by || !by.cases) return row;
    const wi = (row.old_widget_ids || []).indexOf(by.widget);
    const saved = wi >= 0 && Array.isArray(widgetsValues) ? widgetsValues[wi] : undefined;
    if (saved === undefined || !Object.prototype.hasOwnProperty.call(by.cases, String(saved))) return row;
    const { c2c_by_value: _drop, ...base } = row;
    return { ...base, ...by.cases[String(saved)] };
}

/**
 * @param {object} graph - a serialised graph or subgraph definition: { nodes: [...], links: [...] }
 * @param {Map<string, object>} byOld - old_node_id -> replacement row
 * @param {(type: string) => boolean} isRegistered
 * @param {(type: string) => ({inputs: object[], outputs: object[], widgets: string[], values: any[],
 *                             required?: string[]} | null)} describe
 * @returns {{migrated: string[], removed: string[], problems: string[], notes: Map<string, string>}}
 */
export function migrateGraph(graph, byOld, isRegistered, describe) {
    const migrated = [];
    const removed = [];
    const problems = [];
    const notes = new Map();
    if (!graph || !Array.isArray(graph.nodes)) return { migrated, removed, problems, notes };
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
    const gone = new Set();

    for (const node of graph.nodes) {
        const listed = byOld.get(node.type);
        if (!listed || isRegistered(node.type)) continue;       // only types that are really gone
        const row = pickCase(listed, node.widgets_values);

        if (row.c2c_remove) {
            const label = `${node.type} (#${node.id})`;
            for (const inp of Array.isArray(node.inputs) ? node.inputs : []) {
                if (inp.link != null) dropLink(inp.link);
            }
            for (const out of Array.isArray(node.outputs) ? node.outputs : []) {
                if ((out.links || []).length) {
                    problems.push(`${label}: output "${out.name}" fed other nodes; those links were removed`);
                    for (const lid of [...out.links]) dropLink(lid);
                }
            }
            gone.add(node);
            removed.push(row.c2c_note ? `${label} - ${row.c2c_note}` : label);
            continue;
        }

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

        const carryControl = (wi, nw) => {
            if (nw < 0 || info.widgets[nw + 1] !== CONTROL) return;
            if (oldWidgetIds[wi + 1] === CONTROL && oldVals[wi + 1] !== undefined) newVals[nw + 1] = oldVals[wi + 1];
            else newVals[nw + 1] = "fixed";
        };

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
                if (wi >= 0 && nw >= 0 && oldVals[wi] !== undefined) {
                    newVals[nw] = oldVals[wi];
                    carryControl(wi, nw);
                }
            } else if ("set_value" in m) {
                const nw = info.widgets.indexOf(m.new_id);
                if (nw >= 0) newVals[nw] = m.set_value;
            }
        }
        for (const v of row.c2c_values || []) {
            const nw = info.widgets.indexOf(v.new_id);
            if (nw < 0) continue;
            if ("set" in v) { newVals[nw] = v.set; continue; }
            const wi = oldWidgetIds.indexOf(v.old_id);
            if (wi < 0 || oldVals[wi] === undefined) continue;
            const t = translate(v, oldVals[wi]);
            if (t.ok) {
                newVals[nw] = t.value;
            } else {
                newVals[nw] = info.values[nw];                    // input_mapping may have copied it verbatim
                problems.push(`${label}: ${v.old_id} = "${oldVals[wi]}" has no equivalent on ${row.new_node_id}; `
                    + `${v.new_id} left at its default`);
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
        const empty = (info.required || []).filter((name) => {
            const inp = newInputs.find((i) => i.name === name);
            return inp && !inp.widget && inp.link == null;
        });
        if (empty.length) problems.push(`${label}: required input${empty.length > 1 ? "s" : ""} not connected: ${empty.join(", ")}`);

        node.type = row.new_node_id;
        node.inputs = newInputs;
        node.outputs = newOutputs;
        node.widgets_values = newVals;
        if (node.widgets_values_named && typeof node.widgets_values_named === "object") {
            // Comfy.Workflow.NamedValuesRestore restores by name from this map when it exists: keep it in step
            node.widgets_values_named = Object.fromEntries(info.widgets.map((n, i) => [n, newVals[i]]));
        }
        if (node.properties && "Node name for S&R" in node.properties) {
            node.properties["Node name for S&R"] = row.new_node_id;
        }
        if (row.c2c_note) notes.set(`${row.old_node_id} → ${row.new_node_id}`, row.c2c_note);
        migrated.push(label);
    }
    if (gone.size) graph.nodes = graph.nodes.filter((n) => !gone.has(n));
    return { migrated, removed, problems, notes };
}

/**
 * Keep every saved link on its input when a node's input list changed under it (L7.65 P08). The front end builds a
 * node's slots as every socket first, then one slot per widget, and on load (ComfyNode.configure, 1.53.10) re-orders
 * the saved inputs into that layout BY NAME - but leaves each link's target_slot as saved. A survivor that gained a
 * socket (Inpaint Stitch Pro: original_image, mask) therefore moves its widget slots, and a link into one of them
 * (a Primitive on blend_mode_override, say) would point at the wrong slot. Before configure, this rewrites such a
 * node's inputs into the layout configure produces and gives each link the slot its input lands on. Idempotent: a
 * node already in the current layout is left as it is.
 * @param {object} graph - { nodes, links }
 * @param {Set<string>} types - node types to check (registered, current types)
 * @param {(type: string) => ({inputs: object[]} | null)} describe - the type's slots as the front end builds them
 * @returns {string[]} one line per link that moved
 */
export function keepLinkSlots(graph, types, describe) {
    const moved = [];
    if (!graph || !Array.isArray(graph.nodes)) return moved;
    const links = Array.isArray(graph.links) ? graph.links : [];
    const linkById = new Map(links.map((l) => [L.id(l), l]));
    for (const node of graph.nodes) {
        if (!types.has(node.type) || !Array.isArray(node.inputs)) continue;
        const info = describe(node.type);
        if (!info) continue;
        const names = info.inputs.map((i) => i.name);
        const saved = new Map(node.inputs.map((i) => [i.name, i]));
        const layout = [...info.inputs.map((i) => saved.get(i.name) ?? { ...i, link: null }),
                        ...node.inputs.filter((i) => !names.includes(i.name))];          // configure keeps extras last
        if (layout.length === node.inputs.length && layout.every((inp, k) => inp === node.inputs[k])) continue;
        layout.forEach((inp, slot) => {
            if (inp.link == null) return;
            const l = linkById.get(inp.link);
            if (l && L.targetSlot(l) !== slot) {
                moved.push(`${node.type} (#${node.id}) ${inp.name}: slot ${L.targetSlot(l)} -> ${slot}`);
                L.setTargetSlot(l, slot);
            }
        });
        node.inputs = layout;
    }
    return moved;
}

/** keepLinkSlots on the root graph and every subgraph definition. */
export function keepWorkflowLinkSlots(data, types, describe) {
    const moved = [];
    for (const g of [data, ...((data?.definitions?.subgraphs) || [])]) moved.push(...keepLinkSlots(g, types, describe));
    return moved;
}

/** Root graph and every subgraph definition. */
export function migrateWorkflow(data, table, isRegistered, describe) {
    const byOld = new Map();
    for (const row of table || []) if (row?.old_node_id && !byOld.has(row.old_node_id)) byOld.set(row.old_node_id, row);
    const out = { migrated: [], removed: [], problems: [], notes: [] };
    const notes = new Map();
    const graphs = [data, ...((data?.definitions?.subgraphs) || [])];
    for (const g of graphs) {
        const r = migrateGraph(g, byOld, isRegistered, describe);
        out.migrated.push(...r.migrated);
        out.removed.push(...r.removed);
        out.problems.push(...r.problems);
        for (const [k, v] of r.notes) notes.set(k, v);
    }
    out.notes = [...notes].map(([k, v]) => `${k}: ${v}`);
    return out;
}

/**
 * Owner 2026-11-23 ("separate image/latent of load video c2c. always keep separate slots and never mixed"): Load Video
 * (C2C)'s output 0 was "IMAGE,LATENT" (a LATENT when a VAE was wired); it is IMAGE only now and the latent is
 * output 10.
 * type -> {from, to, kind, name, fromType}: a saved link from `from` whose READER takes `kind` (and not IMAGE) moves to
 * `to`; the links that stay on `from` get `fromType` (the saved "IMAGE,LATENT" link type is gone with the mix).
 * Same rule for API prompts: nodes/_legacy_replacements.py OUTPUT_SPLITS - keep the two in step.
 */
export const OUTPUT_SPLITS = { LoadVideoC2C: { from: 0, to: 10, kind: "LATENT", name: "latent", fromType: "IMAGE" } };
const SUBGRAPH_INPUT_ID = -10;
const SUBGRAPH_OUTPUT_ID = -20;
const MAX_DEPTH = 8;

const setLinkType = (l, t) => { if (linkArr(l)) l[5] = t; else l.type = t; };
const sameId = (a, b) => String(a) === String(b);

/** "LATENT" / "IMAGE" when a slot type names exactly one of them; null for a wildcard, a mix or anything else. */
function kindOf(type) {
    const parts = String(type ?? "").split(",").map((s) => s.trim());
    const lat = parts.includes("LATENT"), img = parts.includes("IMAGE");
    return lat && !img ? "LATENT" : img && !lat ? "IMAGE" : null;
}

/** One kind when every reader agrees, else null. */
const agree = (kinds) => (kinds.length && kinds.every((k) => k && k === kinds[0]) ? kinds[0] : null);

/**
 * The kind a link's reader takes: the target input's own type; through a subgraph node into the subgraph (what its
 * inner input feeds); out of a subgraph through its output node (what every instance's output feeds outside).
 */
function readerKind(ctx, graph, link, depth) {
    if (depth > MAX_DEPTH) return null;
    const tid = L.target(link), ts = L.targetSlot(link);
    if (sameId(tid, SUBGRAPH_OUTPUT_ID)) {
        const own = kindOf(graph.outputs?.[ts]?.type);
        if (own || !graph.id) return own;
        const kinds = [];
        for (const g of ctx.graphs) {
            for (const n of g.nodes || []) {
                if (n.type !== graph.id) continue;
                for (const l of g.links || []) {
                    if (!sameId(L.origin(l), n.id) || L.originSlot(l) !== ts) continue;
                    kinds.push(readerKind(ctx, g, l, depth + 1));
                }
            }
        }
        return agree(kinds);
    }
    const node = (graph.nodes || []).find((n) => sameId(n.id, tid));
    if (!node) return null;
    const inputs = Array.isArray(node.inputs) ? node.inputs : [];
    const inp = inputs.find((i) => i && i.link != null && sameId(i.link, L.id(link))) ?? inputs[ts];
    const own = kindOf(inp?.type);
    if (own) return own;
    const def = ctx.defs.get(node.type);
    if (!def) return null;
    const slot = inp ? inputs.indexOf(inp) : ts;
    const inner = (def.links || []).filter((l) => sameId(L.origin(l), SUBGRAPH_INPUT_ID) && L.originSlot(l) === slot);
    return agree(inner.map((l) => readerKind(ctx, def, l, depth + 1)));
}

/**
 * Move each OUTPUT_SPLITS link in one graph: the link's origin_slot and type, and the node's outputs[].links (the
 * outputs list is padded to the new slot: frontend 1.53 configure keeps saved outputs by INDEX and takes each one's
 * name / type from the definition). Idempotent.
 */
function splitGraph(ctx, graph, describe, splits) {
    const moved = [];
    if (!graph || !Array.isArray(graph.nodes)) return moved;
    for (const node of graph.nodes) {
        const sp = splits[node.type];
        if (!sp) continue;
        const onFrom = (graph.links || []).filter((l) => sameId(L.origin(l), node.id) && L.originSlot(l) === sp.from);
        const readers = onFrom.filter((l) => readerKind(ctx, graph, l, 0) === sp.kind);
        for (const l of onFrom) if (!readers.includes(l) && sp.fromType) setLinkType(l, sp.fromType);
        if (!readers.length) continue;
        const outs = Array.isArray(node.outputs) ? node.outputs : (node.outputs = []);
        const defOuts = describe?.(node.type)?.outputs || [];
        for (let i = 0; i <= sp.to; i++) {
            if (outs[i]) continue;
            const d = defOuts[i] || (i === sp.to ? { name: sp.name, type: sp.kind } : { name: "", type: "*" });
            outs[i] = { name: d.name, type: d.type, links: [] };
        }
        const to = outs[sp.to];
        to.links = Array.isArray(to.links) ? to.links : [];
        to.type = sp.kind;
        if (!to.name) to.name = sp.name;
        for (const l of readers) {
            const id = L.id(l);
            L.setOriginSlot(l, sp.to);
            setLinkType(l, sp.kind);
            const from = outs[sp.from];
            if (Array.isArray(from?.links)) from.links = from.links.filter((x) => !sameId(x, id));
            if (!to.links.some((x) => sameId(x, id))) to.links.push(id);
            moved.push(`${node.type} (#${node.id}) link ${id}: output ${sp.from} -> ${sp.to} (${sp.kind})`);
        }
        // a subgraph output fed by this link now carries the split's type
        if (Array.isArray(graph.outputs)) {
            for (const l of readers) {
                if (sameId(L.target(l), SUBGRAPH_OUTPUT_ID) && graph.outputs[L.targetSlot(l)]) {
                    graph.outputs[L.targetSlot(l)].type = sp.kind;
                }
            }
        }
    }
    return moved;
}

/**
 * splitGraph on the root graph and every subgraph definition (the reader of a link may sit across a subgraph
 * boundary, so every graph is visible to every other). Runs after migrateWorkflow (a Load Video Path (C2C) migrated to
 * Load Video (C2C) on output 0 is split too).
 * @param {object} data - the serialised workflow
 * @param {(type: string) => ({outputs: object[]} | null)} [describe]
 * @param {object} [splits] - OUTPUT_SPLITS
 * @returns {string[]} one line per link that moved
 */
export function splitWorkflowOutputs(data, describe = null, splits = OUTPUT_SPLITS) {
    const defsList = (data?.definitions?.subgraphs) || [];
    const graphs = [data, ...defsList].filter(Boolean);
    const ctx = { graphs, defs: new Map(defsList.filter((d) => d?.id).map((d) => [d.id, d])) };
    const moved = [];
    for (const g of graphs) moved.push(...splitGraph(ctx, g, describe, splits));
    return moved;
}
