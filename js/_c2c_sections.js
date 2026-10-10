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
 * Nodes 2.0: core already folds inputs flagged "advanced" behind its own "Show advanced inputs" button (and lists them
 * as ADVANCED INPUTS in the properties panel); the same inputs carry that flag, so there this helper steps aside - no
 * bar, nothing hidden by it - and core's UI is the only one. The classic canvas draws every widget, so the bar is
 * classic-only. Core's "Always show advanced widgets" setting opens every section here too.
 *
 * Setting: C2C › Canvas › Control sections (off = every control the node's rules allow is shown, no bar).
 */
import { app } from "../../scripts/app.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";

const SETTING = "c2c.sections.enabled";
const BAR = "_c2c_sections";
const PROP = "c2c_sections_open";

export function sectionsEnabled() {
    try {
        const get = (id) => app.ui?.settings?.getSettingValue?.(id);
        return get(SETTING) !== false
            && get("Comfy.VueNodes.Enabled") !== true                // Nodes 2.0: core's advanced button owns this
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
    if (!sectionsEnabled()) return true;
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
        if (sectionsEnabled()) {
            for (const name of ownerShown) {
                const s = sectionOf(node, name);
                if (s !== null) counts.set(s, (counts.get(s) || 0) + 1);
            }
        }
        let w = node.widgets?.find((x) => x.name === BAR);
        if (!counts.size) {
            if (w?.element) w.element.style.display = "none";
            if (w) w.computeSize = () => [0, -4];
            return;
        }
        if (!w) {
            const root = document.createElement("div");
            root.className = "c2c-sections-bar";
            Object.assign(root.style, { display: "flex", flexWrap: "wrap", gap: "4px", alignItems: "center",
                padding: "3px 4px", font: "11px system-ui,sans-serif", color: "var(--input-text,#ddd)" });
            w = node.addDOMWidget(BAR, "div", root, { serialize: false });
            w.serialize = false;
        }
        const root = w.element;
        root.style.display = "flex";
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
            Object.assign(chip.style, { font: "inherit", cursor: "pointer", borderRadius: "10px", padding: "1px 8px",
                border: "1px solid rgba(255,255,255,0.25)", color: "inherit",
                background: isOpen ? "rgba(120,140,255,0.28)" : "transparent" });
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
        const rows = Math.ceil(counts.size / 3);
        w.computeSize = (width) => [width, 6 + rows * 22];
    } catch (err) {
        __c2cReport("_c2c_sections", err);
    }
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
