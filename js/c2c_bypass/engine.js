// C2C Bypass Manager engine (L4.03; design docs/research/BYPASS_MANAGER_design.md). Pure functions, no imports, so
// they run under Node in tests. Modes verified in the fe 1.53.10 bundle: ALWAYS 0, NEVER 2 (mute), BYPASS 4.

export const MODE = { NORMAL: 0, MUTE: 2, BYPASS: 4 };
export const MODE_NAME = { 0: "normal", 2: "muted", 4: "bypassed" };

/** Every node of a graph, depth-first through subgraphs: [{node, graph, path: [ids...], depth}]. */
export function walk(graph, path = [], depth = 0, out = [], seen = new Set()) {
    if (!graph || seen.has(graph)) return out;
    seen.add(graph);
    for (const node of graph.nodes ?? graph._nodes ?? []) {
        const p = [...path, node.id];
        out.push({ node, graph, path: p, depth });
        if (node.subgraph) walk(node.subgraph, p, depth + 1, out, seen);
    }
    return out;
}

function typeOf(node) {
    return node.comfyClass ?? node.type ?? "";
}

function titleOf(node) {
    return node.title ?? typeOf(node);
}

function inGroup(node, group) {
    if (!group) return true;
    const b = group._bounding ?? group.bounding ?? group.pos?.concat(group.size ?? []);
    if (!b || b.length < 4) return false;
    const [gx, gy, gw, gh] = b;
    const [x, y] = node.pos ?? [0, 0];
    const [w, h] = node.size ?? [0, 0];
    return x >= gx && y >= gy && x + w <= gx + gw && y + h <= gy + gh;
}

/**
 * rule: {by: "type" | "title" | "regex", value, scope: "everything" | "graph" | "group", graph, group}
 * regex is tested against "type title" (case-insensitive); a bad pattern throws a plain Error.
 */
export function match(entries, rule) {
    const by = rule.by ?? "type";
    let test;
    if (by === "regex") {
        let re;
        try {
            re = new RegExp(String(rule.value ?? ""), "i");
        } catch (e) {
            throw new Error(`Bypass Manager: '${rule.value}' is not a valid pattern (${e.message})`);
        }
        test = (n) => re.test(`${typeOf(n)} ${titleOf(n)}`);
    } else if (by === "title") {
        const v = String(rule.value ?? "").toLowerCase();
        test = (n) => titleOf(n).toLowerCase() === v;
    } else {
        const v = String(rule.value ?? "");
        test = (n) => typeOf(n) === v;
    }
    const scope = rule.scope ?? "everything";
    return entries.filter((e) => {
        if (scope === "graph" && e.graph !== rule.graph) return false;
        if (scope === "group" && (e.graph !== rule.graph || !inGroup(e.node, rule.group))) return false;
        return test(e.node);
    });
}

export function isProtected(node) {
    return Boolean(node.pinned ?? node.flags?.pinned) || Boolean(node.locked ?? node.flags?.locked);
}

/** Set `mode` on the matched nodes. Returns {changed: [entries], skipped: [entries]} (pinned / locked skipped). */
export function apply(entries, mode, { skipProtected = true } = {}) {
    if (![MODE.NORMAL, MODE.MUTE, MODE.BYPASS].includes(mode)) throw new Error(`Bypass Manager: bad mode ${mode}`);
    const changed = [];
    const skipped = [];
    for (const e of entries) {
        if (skipProtected && isProtected(e.node)) {
            skipped.push(e);
            continue;
        }
        if (e.node.mode !== mode) {
            e.node.mode = mode;
            changed.push(e);
        }
    }
    return { changed, skipped };
}

/** Per type: {type, count, normal, muted, bypassed, other, state: "normal" | "muted" | "bypassed" | "mixed"}. */
export function summary(entries) {
    const by = new Map();
    for (const { node } of entries) {
        const t = typeOf(node);
        const s = by.get(t) ?? { type: t, count: 0, normal: 0, muted: 0, bypassed: 0, other: 0 };
        s.count++;
        if (node.mode === MODE.NORMAL || node.mode == null) s.normal++;
        else if (node.mode === MODE.MUTE) s.muted++;
        else if (node.mode === MODE.BYPASS) s.bypassed++;
        else s.other++;
        by.set(t, s);
    }
    for (const s of by.values()) {
        s.state = s.normal === s.count ? "normal" : s.muted === s.count ? "muted"
            : s.bypassed === s.count ? "bypassed" : "mixed";
    }
    return [...by.values()].sort((a, b) => a.type.localeCompare(b.type));
}

/** Next state when the user clicks a type's tri-state button: normal -> bypassed -> muted -> normal (mixed -> normal). */
export function nextMode(state) {
    return state === "normal" ? MODE.BYPASS : state === "bypassed" ? MODE.MUTE : MODE.NORMAL;
}

// ── named sets, saved in the workflow (graph.extra.c2c_bypass_sets) ─────────────────────────────────────────────

export function saveSet(rootGraph, name, entries) {
    rootGraph.extra = rootGraph.extra ?? {};
    const sets = (rootGraph.extra.c2c_bypass_sets = rootGraph.extra.c2c_bypass_sets ?? {});
    sets[name] = entries.map((e) => ({ path: e.path.join(":"), mode: e.node.mode ?? MODE.NORMAL }));
    return sets[name].length;
}

export function applySet(rootGraph, name, opts = {}) {
    const set = rootGraph.extra?.c2c_bypass_sets?.[name];
    if (!set) throw new Error(`Bypass Manager: no saved set named '${name}'`);
    const byPath = new Map(walk(rootGraph).map((e) => [e.path.join(":"), e]));
    const changed = [];
    const missing = [];
    const skipped = [];
    for (const { path, mode } of set) {
        const e = byPath.get(path);
        if (!e) {
            missing.push(path);
            continue;
        }
        const r = apply([e], mode, opts);
        changed.push(...r.changed);
        skipped.push(...r.skipped);
    }
    return { changed, skipped, missing };
}
