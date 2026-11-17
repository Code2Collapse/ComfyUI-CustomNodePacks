/**
 * A28 (ORDERS 29) node layout engine: Python declares which inputs show when, this one module applies it to every node
 * that declares it, in classic LiteGraph AND Nodes 2.0, also inside subgraphs. Contract: docs/audit/A28/README.md.
 *
 * Option keys on an input spec (V1 options dict / V3 extra_dict; core passes them to nodeData.input untouched):
 *   c2c_section          "Tracking"                  the input lives in a collapsible section, closed by default
 *   c2c_show_if          {"face_finder": ["yunet"]}  shown only while EVERY listed widget matches its condition: a
 *                                                    list (one of), {"not": [...]}, {"gt"|"gte"|"lt"|"lte": n},
 *                                                    {"truthy": true|false} (js/_c2c_layout_rules.js matchValue)
 *   c2c_show_if_any      [{...}, {...}]              shown while ANY group (each an AND like c2c_show_if) matches
 *   c2c_hide_if_linked   ["face_mask"]               hidden while any listed input socket is connected
 *   c2c_show_if_linked   ["control_image"]           shown only while ANY listed input socket is connected
 *   c2c_hidden           true                        never shown (LEGACY / INTERNAL / DEAD), still saved and sent
 *   c2c_show_if_status   "model_missing"             shown only while GET /c2c/node_status?type=<type> says so;
 *                                                    a type without a status provider (404) never matches, an
 *                                                    unreachable / failing route shows the control (honest default)
 *
 * Visibility never changes execution: a hidden widget keeps its value, is saved and goes into the prompt. A widget
 * whose OWN socket is linked is never hidden (a wire nobody can see or undo).
 *
 * Hide = no space. Classic: `widget.hidden` (fe 1.53.10 LGraphNode.isWidgetVisible / getLayoutWidgets / computeSize
 * skip it) via the shared setWidgetVisible, which also hides a DOM widget's element. Nodes 2.0: `widget.hidden` is
 * `options.hidden` in the reactive widget store (BaseWidget.hidden -> _state.options.hidden), which
 * processedWidgetRenderModel.isWidgetVisible reads, so the row disappears; vueSyncNodeWidgets nudges the row rebuild.
 * Resize after every change: classic setSize([max(user width, computed min), computeSize()[1]]); Nodes 2.0
 * vueFitNodeHeight (the content height; Nodes 2.0 grows a node by itself but never shrinks it). No fixed heights.
 *
 * Sections, both renderers: a "More" bar (one non-serialised DOM widget, appended LAST so saved widget positions
 * never shift - fe 1.53 restores widgets_values by position) with one chip per section that holds a control the rules
 * currently show. Open sections per node in node.properties.c2c_sections_open (saved with the workflow). Inputs in our
 * sections must not carry core's "advanced" flag (Nodes 2.0 would fold them twice): a node type that combines the two
 * is the older pattern owned by js/c2c_mode_widgets.js / js/mask_visibility.js / js/motion_mask_tracker.js, and this
 * engine leaves it alone. Another per-node visibility owner can opt a type out:
 * globalThis.__c2cVisibilityOwners.add(type) - one visibility owner per node.
 *
 * Settings honoured: C2C > Canvas > Control sections (off = every section open, no bar) and core's
 * Comfy.Node.AlwaysShowAdvancedWidgets (on = every section open).
 *
 * API: relayoutNode(node) (re-evaluate + resize now), layoutRules(type), widgetShown(node, name).
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { setWidgetVisible, vueFitNodeHeight, vueNodesActive, vueSyncNodeWidgets } from "./_widget_visibility.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";
import { parseInputRule, ruleAllows, watchedBy } from "./_c2c_layout_rules.js";

const BAR = "_c2c_layout_sections";
const PROP = "c2c_sections_open";
const SETTING = "c2c.sections.enabled";
const BAR_MARGIN = 4;
const CHIP_FONT = "11px system-ui, sans-serif";
const CHIP_H = 20;                 // line-height 16 + padding 1+1 + border 1+1
const GAP = 4;
const PAD_Y = 3, PAD_X = 4;

const RULES = new Map();           // node type -> { rules: Map(name -> rule), watch: Set, status: bool } | null
const LIVE = new Set();            // managed nodes currently in a graph (event-driven refresh only, never polled)
const STATUS = new Map();          // node type -> { state: "pending" | "ok" | "failed", data }

globalThis.__c2cVisibilityOwners = globalThis.__c2cVisibilityOwners || new Set();

// ── spec ─────────────────────────────────────────────────────────────────────────────────────────────────────────

function parseRules(nodeData) {
    const input = nodeData?.input || {};
    const rules = new Map();
    let legacy = false;
    for (const group of ["required", "optional"]) {
        for (const [name, spec] of Object.entries(input[group] || {})) {
            const o = Array.isArray(spec) && spec[1] && typeof spec[1] === "object" ? spec[1] : null;
            const r = parseInputRule(o);
            if (!r) continue;
            if (r.section && o.advanced) legacy = true;
            rules.set(name, r);
        }
    }
    if (!rules.size || legacy) return null;
    const watch = new Set();
    let status = false;
    for (const r of rules.values()) {
        for (const k of watchedBy(r)) watch.add(k);
        if (r.showIfStatus) status = true;
    }
    return { rules, watch, status };
}

export function layoutRules(type) {
    return RULES.get(type) || null;
}

function managed(node) {
    const spec = RULES.get(node?.type);
    return spec && !globalThis.__c2cVisibilityOwners.has(node.type) ? spec : null;
}

// ── state readers ────────────────────────────────────────────────────────────────────────────────────────────────

function setting(id) {
    try { return app.ui?.settings?.getSettingValue?.(id); } catch (_e) { return undefined; }
}

function sectionsOn() {
    return setting(SETTING) !== false && setting("Comfy.Node.AlwaysShowAdvancedWidgets") !== true;
}

function openSet(node) {
    const v = node.properties?.[PROP];
    return new Set(Array.isArray(v) ? v : []);
}

function widgetValue(node, name) {
    return node.widgets?.find((w) => w.name === name)?.value;
}

function socketLinked(node, name) {
    const i = node.inputs?.find((s) => s.name === name);
    return i != null && i.link != null;
}

function ownSocketLinked(node, w) {
    const i = node.inputs?.find((s) => s.widget?.name === w.name);
    return i != null && i.link != null;
}

function statusSays(type, key) {
    const s = STATUS.get(type);
    if (!s || s.state === "pending") return false;           // no flash while the first answer is on its way
    if (s.state === "none") return false;                    // no provider for this type: never matches
    if (s.state === "failed") return true;                   // unknown: show rather than hide a needed control
    return s.data?.[key] === true;
}

/** True when the rules (not the sections) want this input shown. */
function ruled(node, r) {
    return ruleAllows(r, { get: (k) => widgetValue(node, k), linked: (k) => socketLinked(node, k),
                           status: (k) => statusSays(node.type, k) });
}

/** Whether `name` is shown right now (rules + its section). */
export function widgetShown(node, name) {
    const spec = managed(node);
    const r = spec?.rules.get(name);
    if (!r) return true;
    return ruled(node, r) && (!r.section || !sectionsOn() || openSet(node).has(r.section));
}

// ── model status ─────────────────────────────────────────────────────────────────────────────────────────────────

function fetchStatus(type) {
    STATUS.set(type, { state: STATUS.get(type)?.state === "ok" ? "ok" : "pending", data: STATUS.get(type)?.data });
    api.fetchApi(`/c2c/node_status?type=${encodeURIComponent(type)}`)
        .then((r) => (r.status === 404 ? { none: true } : r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
        .then((j) => STATUS.set(type, j?.none ? { state: "none", data: null } : { state: "ok", data: j?.status || {} }))
        .catch(() => STATUS.set(type, { state: "failed", data: null }))
        .finally(() => { for (const n of LIVE) if (n.type === type) schedule(n); });
}

function ensureStatus(type) {
    if (!STATUS.has(type)) fetchStatus(type);
}

function refreshAllStatus() {
    const types = new Set([...LIVE].filter((n) => RULES.get(n.type)?.status).map((n) => n.type));
    for (const t of types) fetchStatus(t);
}

// ── section bar ──────────────────────────────────────────────────────────────────────────────────────────────────

let _measureCtx = null;
function textWidth(s) {
    try {
        _measureCtx = _measureCtx || document.createElement("canvas").getContext("2d");
        _measureCtx.font = CHIP_FONT;
        return _measureCtx.measureText(s).width;
    } catch (_e) { return s.length * 6; }
}

const chipText = (name, n, open) => `${name} ${n} ${open ? "▾" : "▸"}`;

/** Content height of the bar at a node width: the same flex-wrap the element does, from measured text widths. */
function barHeight(node, width) {
    const chips = node.__c2cLayoutChips || [];
    if (!chips.length) return 0;
    const avail = Math.max(40, width - 2 * BAR_MARGIN - 2 * PAD_X);
    let rows = 1;
    let x = Math.ceil(textWidth("More:")) + GAP;
    for (const c of chips) {
        const w = Math.ceil(textWidth(chipText(c.name, c.count, c.open))) + 18;
        if (x > 0 && x + w > avail) { rows += 1; x = 0; }
        x += w + GAP;
    }
    return 2 * PAD_Y + rows * CHIP_H + (rows - 1) * GAP;
}

function ensureBar(node) {
    let w = node.widgets?.find((x) => x.name === BAR);
    if (w) return w;
    const root = document.createElement("div");
    root.className = "c2c-sections-bar c2c-layout-sections";
    Object.assign(root.style, { display: "flex", flexWrap: "wrap", gap: `${GAP}px`, alignItems: "center",
        alignContent: "flex-start", padding: `${PAD_Y}px ${PAD_X}px`, boxSizing: "border-box",
        font: CHIP_FONT, lineHeight: "16px", color: "var(--input-text, #ddd)" });
    w = node.addDOMWidget(BAR, "c2c-sections", root, { serialize: false, margin: BAR_MARGIN });
    w.serialize = false;                                     // serialiseWidgetValues checks w.serialize, not options
    w.computeSize = (width) => {
        const W = Math.max(width || 0, node.__c2cLayoutWidth || node.size?.[0] || 0);
        return [W, barHeight(node, W) + 2 * BAR_MARGIN];
    };
    return w;
}

function drawBar(node, bar) {
    const root = bar.element;
    root.replaceChildren();
    const label = document.createElement("span");
    label.textContent = "More:";
    label.style.opacity = ".7";
    root.append(label);
    for (const c of node.__c2cLayoutChips) {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "c2c-section-chip";
        chip.dataset.section = c.name;
        chip.textContent = chipText(c.name, c.count, c.open);
        chip.title = c.open ? `Hide the ${c.name} controls` : `Show the ${c.name} controls (${c.count})`;
        chip.setAttribute("aria-expanded", String(c.open));
        Object.assign(chip.style, { font: "inherit", lineHeight: "16px", cursor: "pointer", borderRadius: "10px",
            padding: "1px 8px", margin: "0", whiteSpace: "nowrap", border: "1px solid rgba(255,255,255,0.25)",
            color: "inherit", background: c.open ? "rgba(120,140,255,0.28)" : "transparent" });
        chip.addEventListener("pointerdown", (e) => e.stopPropagation());
        chip.addEventListener("click", (e) => {
            e.stopPropagation();
            toggleSection(node, c.name);
        });
        root.append(chip);
    }
}

export function toggleSection(node, name) {
    const now = openSet(node);
    if (now.has(name)) now.delete(name); else now.add(name);
    node.properties = node.properties || {};
    node.properties[PROP] = [...now];
    relayoutNode(node);
}

// ── apply ────────────────────────────────────────────────────────────────────────────────────────────────────────

function show(w, want) {
    if (want === (w.__mec_hidden !== true)) return false;
    setWidgetVisible(w, want);
    for (const lw of w.linkedWidgets || []) setWidgetVisible(lw, want);   // a seed's control_after_generate
    return true;
}

/** Re-evaluate every declared input of `node` and resize it. */
export function relayoutNode(node) {
    const spec = managed(node);
    if (!spec || !node.widgets) return;
    if (spec.status) ensureStatus(node.type);
    const sectionsActive = sectionsOn();
    const open = openSet(node);
    const counts = new Map();
    let changed = false;
    for (const w of node.widgets) {
        const r = spec.rules.get(w.name);
        if (!r) continue;
        let want = ruled(node, r);
        if (want && r.section && sectionsActive) {
            counts.set(r.section, (counts.get(r.section) || 0) + 1);
            want = open.has(r.section);
        }
        if (ownSocketLinked(node, w)) want = true;
        changed = show(w, want) || changed;
    }
    const chips = [...counts].map(([name, count]) => ({ name, count, open: open.has(name) }));
    const sig = JSON.stringify(chips);
    if (chips.length || node.widgets.some((x) => x.name === BAR)) {
        const bar = ensureBar(node);
        if (sig !== node.__c2cLayoutSig) {
            node.__c2cLayoutChips = chips;
            node.__c2cLayoutSig = sig;
            drawBar(node, bar);
            changed = true;
        }
        changed = show(bar, chips.length > 0) || changed;
    }
    if (changed || !node.__c2cLayoutFitted) fit(node);
}

function fit(node) {
    node.__c2cLayoutFitted = true;
    if (vueNodesActive()) {
        vueSyncNodeWidgets(node);
        vueFitNodeHeight(node);
        return;
    }
    try {
        const min = node.computeSize();
        const width = Math.max(node.size?.[0] || 0, min[0]);
        node.__c2cLayoutWidth = width;
        const h = node.computeSize()[1];
        node.setSize([width, h]);
    } finally {
        node.__c2cLayoutWidth = undefined;
    }
    node.setDirtyCanvas?.(true, true);
}

function schedule(node) {
    if (!node || node.__c2cLayoutPending) return;
    node.__c2cLayoutPending = true;
    queueMicrotask(() => {
        node.__c2cLayoutPending = false;
        try { if (node.graph) relayoutNode(node); } catch (err) { __c2cReport("_c2c_node_layout", err); }
    });
}

function hookWidgets(node, spec) {
    for (const w of node.widgets || []) {
        if (!spec.watch.has(w.name) || w.__c2cLayoutHooked) continue;
        w.__c2cLayoutHooked = true;
        const cb = w.callback;
        w.callback = function (...a) {
            const r = cb?.apply(this, a);
            schedule(node);
            return r;
        };
    }
}

function attach(node) {
    const spec = managed(node);
    if (!spec) return;
    LIVE.add(node);
    hookWidgets(node, spec);
    schedule(node);
}

if (!globalThis.__c2cNodeLayoutExt) {
    globalThis.__c2cNodeLayoutExt = true;
    app.registerExtension({
        name: "C2C.NodeLayout",
        beforeRegisterNodeDef(nodeType, nodeData) {
            const spec = parseRules(nodeData);
            RULES.set(nodeData?.name, spec);
            if (!spec) return;
            const proto = nodeType.prototype;
            const wrap = (key, after) => {
                const orig = proto[key];
                proto[key] = function (...a) {
                    const r = orig?.apply(this, a);
                    try { after.call(this, ...a); } catch (err) { __c2cReport("_c2c_node_layout", err); }
                    return r;
                };
            };
            wrap("onNodeCreated", function () { queueMicrotask(() => attach(this)); });
            wrap("onConfigure", function () { attach(this); });            // values are restored before onConfigure
            wrap("onAfterGraphConfigured", function () { attach(this); });
            wrap("onAdded", function () { attach(this); });                  // paste, convert to subgraph, undo
            wrap("onConnectionsChange", function () { if (managed(this)) schedule(this); });
            wrap("onRemoved", function () { LIVE.delete(this); });
            wrap("onResize", function () {
                // a width drag can re-wrap the bar: follow it once the drag frame is done (classic only)
                if (vueNodesActive() || !this.__c2cLayoutChips?.length) return;
                const bar = this.widgets?.find((x) => x.name === BAR);
                if (!bar || bar.hidden) return;
                const h = barHeight(this, this.size[0]);
                if (h === this.__c2cLayoutBarH) return;
                this.__c2cLayoutBarH = h;
                requestAnimationFrame(() => { this.__c2cLayoutFitted = false; schedule(this); });
            });
        },
        setup() {
            // a model download (or a missing-model error) changes what c2c_show_if_status shows
            api.addEventListener("execution_success", refreshAllStatus);
            api.addEventListener("execution_error", refreshAllStatus);
            // entering / leaving a subgraph mounts that graph's nodes (Nodes 2.0 re-measures them)
            const canvasEl = app.canvas?.canvas;
            canvasEl?.addEventListener?.("litegraph:set-graph", (e) => {
                const g = e?.detail?.newGraph;
                for (const n of LIVE) {
                    if (!n.graph) { LIVE.delete(n); continue; }
                    if (n.graph === g) { n.__c2cLayoutFitted = false; schedule(n); }
                }
            });
        },
    });
}
