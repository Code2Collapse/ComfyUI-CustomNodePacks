/**
 * A28 declarative visibility rules: the pure part of js/_c2c_node_layout.js (no frontend imports, so it is unit-tested
 * under plain Node: tests/test_a28_layout_rules.py). Contract: docs/audit/A28/README.md section 2.
 */

export const KEYS = ["c2c_section", "c2c_show_if", "c2c_show_if_any", "c2c_hide_if_linked", "c2c_show_if_linked",
                     "c2c_hidden", "c2c_show_if_status"];

const isObj = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const num = (v) => (typeof v === "number" ? v : typeof v === "boolean" ? Number(v) : Number(String(v)));

/**
 * One widget value against one condition.
 *   ["a", "b"]           one of (compared as strings)
 *   {"not": ["a"]}       none of
 *   {"gt": n} {"gte": n} {"lt": n} {"lte": n}   numeric
 *   {"truthy": true}     truthy (false: falsy)
 *   {"re": "pattern"}    the value as text matches the pattern (case-insensitive; a bad pattern never matches)
 *   scalar               equal (as strings)
 * Several operators in one object must all hold. An unknown operator never matches (fail closed, so a typo hides
 * instead of silently showing).
 */
export function matchValue(value, cond) {
    if (Array.isArray(cond)) return cond.map(String).includes(String(value));
    if (!isObj(cond)) return String(value) === String(cond);
    let any = false;
    for (const [op, arg] of Object.entries(cond)) {
        any = true;
        if (op === "not") {
            if ((Array.isArray(arg) ? arg : [arg]).map(String).includes(String(value))) return false;
        } else if (op === "gt" || op === "gte" || op === "lt" || op === "lte") {
            const v = num(value), a = num(arg);
            if (!Number.isFinite(v) || !Number.isFinite(a)) return false;
            if (op === "gt" && !(v > a)) return false;
            if (op === "gte" && !(v >= a)) return false;
            if (op === "lt" && !(v < a)) return false;
            if (op === "lte" && !(v <= a)) return false;
        } else if (op === "truthy") {
            const t = !(value === false || value === 0 || value === "" || value == null || value === "false");
            if (t !== Boolean(arg)) return false;
        } else if (op === "re") {
            // A28 CNP part A: a case-insensitive pattern on the value as text (Load Video: the source's extension)
            let re;
            try { re = new RegExp(String(arg), "i"); } catch (_e) { return false; }
            if (!re.test(value == null ? "" : String(value))) return false;
        } else {
            return false;
        }
    }
    return any;
}

/** Every widget of the group matches (AND). `get(name)` reads a widget value. */
export function matchGroup(group, get) {
    return Object.entries(group || {}).every(([k, c]) => matchValue(get(k), c));
}

/** The rule object for one input's options (null when it declares none of the keys). */
export function parseInputRule(o) {
    if (!isObj(o) || !KEYS.some((k) => k in o)) return null;
    const r = {};
    if (o.c2c_section) r.section = String(o.c2c_section);
    if (isObj(o.c2c_show_if)) r.showIf = o.c2c_show_if;
    if (Array.isArray(o.c2c_show_if_any)) r.showIfAny = o.c2c_show_if_any.filter(isObj);
    if (Array.isArray(o.c2c_hide_if_linked)) r.hideIfLinked = o.c2c_hide_if_linked.map(String);
    if (Array.isArray(o.c2c_show_if_linked)) r.showIfLinked = o.c2c_show_if_linked.map(String);
    if (o.c2c_hidden === true) r.hidden = true;
    if (o.c2c_show_if_status) r.showIfStatus = String(o.c2c_show_if_status);
    return r;
}

/** The widgets whose values a rule reads (their changes re-run the layout). */
export function watchedBy(r) {
    const out = new Set(Object.keys(r.showIf || {}));
    for (const g of r.showIfAny || []) for (const k of Object.keys(g)) out.add(k);
    return out;
}

/**
 * True when the rules (not the sections) want the input shown.
 * ctx = { get(name) -> widget value, linked(socket) -> bool, status(key) -> bool }
 */
export function ruleAllows(r, ctx) {
    if (r.hidden) return false;
    if (r.showIf && !matchGroup(r.showIf, ctx.get)) return false;
    if (r.showIfAny && r.showIfAny.length && !r.showIfAny.some((g) => matchGroup(g, ctx.get))) return false;
    for (const s of r.hideIfLinked || []) if (ctx.linked(s)) return false;
    if (r.showIfLinked && !r.showIfLinked.some((s) => ctx.linked(s))) return false;
    if (r.showIfStatus && !ctx.status(r.showIfStatus)) return false;
    return true;
}
