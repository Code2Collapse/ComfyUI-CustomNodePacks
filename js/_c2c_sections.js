/**
 * Collapsible control sections for big C2C nodes (L7.81; owner 2026-10-10 P05, ORDERS 11.4).
 *
 * Membership is declared in Python next to each input: an option key "c2c_section": "<name>". Core passes unknown
 * option keys through to nodeData.input, and nothing else reads this one.
 *
 * This helper never hides anything on its own: each node's ONE visibility owner (js/c2c_mode_widgets.js,
 * js/mask_visibility.js, ...) asks it which of the controls it wants shown sit in a closed section, so two scripts
 * never fight over one widget. The section bar is a single DOM widget appended LAST: frontend 1.52.7 saves
 * widgets_values at each widget's index among ALL widgets but restores them skipping non-serialized ones, so a
 * non-serialized widget placed among serialized ones would shift every later value on reload (measured,
 * docs/evidence/L7.81). Widgets themselves never move; a closed section just hides its members.
 *
 * Nodes 2.0: a node whose sectioned inputs still carry core's "advanced" flag is folded by core's own "Show advanced
 * inputs" button, so there this helper steps aside for it (no bar, nothing hidden by it). A28 (ORDERS 29): a node
 * whose sectioned inputs dropped that flag (MaskOps, Save Video) gets the same bar in Nodes 2.0 as on the classic
 * canvas - Nodes 2.0 would otherwise fold them twice, or not at all. Core's "Always show advanced widgets" setting
 * opens every section here too.
 *
 * Setting: C2C › Canvas › Control sections (off = every control the node's rules allow is shown, no bar).
 */
import { app } from "../../scripts/app.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";

const SETTING = "c2c.sections.enabled";
const BAR = "_c2c_sections";
const PROP = "c2c_sections_open";

/** True when a sectioned input of this node type also carries core's "advanced" flag (the pre-A28 pattern). */
function coreFolds(node) {
    const input = LiteGraph.registered_node_types?.[node?.type]?.nodeData?.input || {};
    for (const g of [input.required, input.optional]) {
        for (const spec of Object.values(g || {})) {
            const o = Array.isArray(spec) && spec[1] && typeof spec[1] === "object" ? spec[1] : null;
            if (o?.c2c_section && o.advanced) return true;
        }
    }
    return false;
}

export function sectionsEnabled(node) {
    try {
        const get = (id) => app.ui?.settings?.getSettingValue?.(id);
        return get(SETTING) !== false
            // Nodes 2.0: core's advanced button owns a node whose sectioned inputs carry its flag
            && !(get("Comfy.VueNodes.Enabled") === true && (!node || coreFolds(node)))
            && get("Comfy.Node.AlwaysShowAdvancedWidgets") !== true;
    } catch (_e) { return true; }
}

/** The section a widget belongs to, from the node definition (null = always shown). */
export function sectionOf(node, name) {
    const input = LiteGraph.registered_node_types?.[node.type]?.nodeData?.input || {};
    const spec = input.required?.[name] || input.optional?.[name];
    const opts = Array.isArray(spec) && spec.length > 1 && spec[1] && typeof spec[1] === "object" ? spec[1] : null;
    return opts?.c2c_section ? String(opts.c2c_section) : null;
}

function openSet(node) {
    const v = node.properties?.[PROP];
    return new Set(Array.isArray(v) ? v : []);
}

/** False when the widget sits in a closed section (and sections are on). */
export function sectionOpen(node, name) {
    if (!sectionsEnabled(node)) return true;
    const s = sectionOf(node, name);
    return s === null || openSet(node).has(s);
}

/**
 * Draw (or refresh) the bar from the controls the owner wants shown: one chip per section that holds at least one
 * of them, with how many. `reapply` re-runs the owner after a chip is clicked.
 */
export function updateSectionBar(node, ownerShown, reapply) {
    try {
        const counts = new Map();
        if (sectionsEnabled(node)) {
            for (const name of ownerShown) {
                const s = sectionOf(node, name);
                if (s !== null) counts.set(s, (counts.get(s) || 0) + 1);
            }
        }
        let w = node.widgets?.find((x) => x.name === BAR);
        if (!counts.size) {
            if (w?.element) w.element.style.display = "none";
            if (w) {
                w.computeSize = () => [0, -4];
                w.hidden = true;                                     // Nodes 2.0 reads options.hidden
                w.options = w.options || {};
                w.options.hidden = true;
            }
            return;
        }
        if (!w) {
            const root = document.createElement("div");
            root.className = "c2c-sections-bar";
            Object.assign(root.style, { display: "flex", flexWrap: "wrap", gap: "4px", alignItems: "center",
                padding: "3px 4px", font: "11px system-ui,sans-serif", lineHeight: "16px", boxSizing: "border-box",
                alignContent: "flex-start", color: "var(--input-text,#ddd)" });
            w = node.addDOMWidget(BAR, "div", root, { serialize: false, margin: 4 });
            w.serialize = false;
        }
        const root = w.element;
        root.style.display = "flex";
        w.hidden = false;
        if (w.options) w.options.hidden = false;
        const open = openSet(node);
        root.replaceChildren();
        const label = document.createElement("span");
        label.textContent = "More:";
        label.style.opacity = ".7";
        root.append(label);
        for (const [name, n] of counts) {
            const chip = document.createElement("button");
            chip.type = "button";
            const isOpen = open.has(name);
            chip.textContent = `${name} ${n} ${isOpen ? "▾" : "▸"}`;
            chip.title = isOpen ? `Hide the ${name} controls` : `Show the ${name} controls`;
            chip.setAttribute("aria-expanded", String(isOpen));
            Object.assign(chip.style, { font: "inherit", lineHeight: "16px", margin: "0", whiteSpace: "nowrap",
                cursor: "pointer", borderRadius: "10px", padding: "1px 8px",
                border: "1px solid rgba(255,255,255,0.25)", color: "inherit",
                background: isOpen ? "rgba(120,140,255,0.28)" : "transparent" });
            chip.dataset.section = name;
            chip.className = "c2c-section-chip";
            chip.addEventListener("pointerdown", (e) => e.stopPropagation());   // a click, not a node drag
            chip.addEventListener("click", (e) => {
                e.stopPropagation();
                const now = openSet(node);
                if (now.has(name)) now.delete(name); else now.add(name);
                node.properties = node.properties || {};
                node.properties[PROP] = [...now];
                try { reapply(); } catch (err) { __c2cReport("_c2c_sections", err); }
            });
            root.append(chip);
        }
        // A28: the height the chips really wrap to at this width (measured text, like js/_c2c_node_layout.js)
        const chips = [...counts].map(([name, n]) => `${name} ${n} ${open.has(name) ? "▾" : "▸"}`);
        w.computeSize = (width) => [width, barHeight(chips, width || node.size?.[0] || 300) + 8];   // + margin 4 + 4
    } catch (err) {
        __c2cReport("_c2c_sections", err);
    }
}

let _ctx = null;
function textWidth(t) {
    try {
        _ctx = _ctx || document.createElement("canvas").getContext("2d");
        _ctx.font = "11px system-ui, sans-serif";
        return _ctx.measureText(t).width;
    } catch (_e) { return t.length * 6; }
}

/** Height of a flex-wrap line of chips (padding 3/4 px, gap 4, chip = text + 18 px wide, 16 + 2 + 2 px tall). */
function barHeight(chips, width) {
    const avail = Math.max(40, width - 8 - 20);
    let rows = 1;
    let x = Math.ceil(textWidth("More:")) + 4;
    for (const c of chips) {
        const cw = Math.ceil(textWidth(c)) + 18;
        if (x > 0 && x + cw > avail) { rows += 1; x = 0; }
        x += cw + 4;
    }
    return 6 + rows * 20 + (rows - 1) * 4;
}

if (!globalThis.__c2cSectionsExt) {
    globalThis.__c2cSectionsExt = true;
    app.registerExtension({
        name: "C2C.Sections",
        settings: [{
            id: SETTING,
            name: "Control sections",
            tooltip: "Big C2C nodes keep their advanced controls in sections you open from a 'More' bar. Off = "
                + "every control the node's current mode uses is shown at once.",
            type: "boolean",
            defaultValue: true,
            category: ["c2c", "Canvas", "Sections"],
        }],
    });
}
