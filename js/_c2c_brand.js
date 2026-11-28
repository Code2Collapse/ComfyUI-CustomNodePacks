/**
 * _c2c_brand.js — the house look on the canvas: night title bar, night body,
 * moonlight title text, on every node THIS pack registers.
 *
 * One copy ships in every Code2Collapse pack (they cannot import each other: a
 * cross-pack import 404s on a standalone install and a 404 on a module import
 * takes the whole pack's front-end down). The copies are identical except for
 * the depth of the `app.js` import; CustomNodePacks' tests hold them in step.
 *
 * WHICH NODES: this pack's own, and only those. The folder this file was
 * served from (`import.meta.url` -> /extensions/<folder>/...) is compared with
 * each node's `python_module` ("custom_nodes.<folder>"). Both come from the
 * same install directory, so it is right whatever the folder is called - the
 * Linux install uses the GitHub names (ComfyUI-NukeNodePack,
 * ComfyUI-MiniMax-H3-Suite), this workspace uses others. A node with no
 * python_module is left alone rather than guessed at.
 *
 * WHO WINS: the user. Colours are applied only when the node has none, so a
 * colour from a saved workflow (restored before onNodeCreated runs) or from
 * the right-click Colors menu is never overwritten. Setting
 * "C2C -> Brand node colours" turns it off.
 *
 * Colours are literal hex on purpose: node colours are painted on the canvas,
 * and a canvas cannot parse var() - it silently paints black.
 *
 * FIT: a node is also grown to its own minimum size (computeSize), one frame
 * after it is created. DOM widgets (status strips, scopes, editors) are added
 * in onNodeCreated, after the node was sized, so fresh nodes came out 8-19px
 * shorter than their content with the bottom row clipped (275 nodes measured
 * across four packs), and small ones narrower than their own title
 * ("Invert (Nuke..."). Only ever up, to the size LiteGraph would enforce on the
 * first drag anyway: a size the user made larger, or a saved one that fits, is
 * kept. Only nodes that carry such a DOM widget - the measured cause - are
 * touched; a node that is small on purpose (a reroute dot) is left alone.
 */

import { app } from "../../scripts/app.js";

// Night node style. History: the first body (#17182f) was a dark fill with no
// edge and vanished into ComfyUI's canvas; lifting it to indigo (#282a56) made
// it visible but "full purple, no darkness". Now the body stays dark and the
// EDGE carries the node (since 2026-11-28 NIGHTCALL: one 1px rim + the wolf disc, no glow, spine or aurora) in the
// pack's colour, plus the wolf badge. `title` is the flat equivalent of the
// title gradient, used where only one colour can be given (collapsed nodes,
// menus, the Nodes 2.0 wrapper).
export const BRAND = Object.freeze({
    // NIGHTCALL (owner-approved 2026-11-28, docs/agy_review/PALETTE_NIGHTCALL.md): every shade is a step of the
    // nightcall ramp in _c2c_theme.js - a cold blue-black, darker than ComfyUI's canvas. What separates a C2C node
    // is a crisp 1px rim in the pack's colour and the wolf disc; no glow, no aurora, no bloom.
    title: "#10141d",           // nightcall bg: collapsed nodes, menus, Nodes 2.0 header
    body: "#0c0f16",            // nightcall bg2
    ink: "#e6e9f0",             // nightcall fg: moonlight silver
    ink2: "#b0b8c7",            // nightcall sub
    titleTop: "#10141d",        // flat header (top == bottom: no gradient)
    titleBottom: "#10141d",
    badge: "#07090d",           // nightcall bg3: wolf badge disc
    widgetBg: "#07090d",        // nightcall bg3: fields sit below the body
    widgetOutline: "#232938",   // nightcall surface1: a quiet NEUTRAL field rim (never pack-tinted)
    widgetText: "#e6e9f0",      // nightcall fg
    widgetText2: "#788299",     // nightcall overlay2
});

// Colours this file handed out before. A workflow saved since then stores
// them on each node; they are ours, not a choice the user made, so they are
// upgraded rather than preserved.
// + the night-era (indigo) colours, so a workflow saved then moves to nightcall; a colour the user picked still wins
const LEGACY_TITLES = new Set(["#26275a", "#33357c", "#11122b", "#12132f"]);
const LEGACY_BODIES = new Set(["#17182f", "#282a56", "#0b0c1d", "#0c0d23"]);

// One restrained accent per pack (NIGHTCALL, a night drive): the 1px rim, the wolf disc ring, focus and active
// controls inside the node. Dark-theme values; PACK_STRIPE_LIGHT is the same hue for a light ground (>= 5.16:1).
const PACK_STRIPE = [
    ["nukemax", "#c28352"], ["nukenode", "#c28352"],              // sodium streetlamp
    ["ltxenhanced", "#c9666c"], ["ltxe", "#c9666c"],              // distant tail-light
    ["minimax", "#5da39d"],                                       // dashboard petrol
    ["wannodeexperiments", "#6f89c4"],                            // mercury cobalt
    ["wananimatepreprocess", "#9b7eb8"],                          // dusk amethyst
    ["wananimalpreprocess", "#79a37b"],                           // canyon pine
    ["glm", "#b5a05e"],                                           // halogen brass
    ["omniscale", "#b87896"], ["magnific", "#b87896"],            // side-mirror rose
];
export const PACK_STRIPE_LIGHT = Object.freeze({
    "#78aec9": "#285e78", "#c28352": "#8c4e1f", "#c9666c": "#9c2d35", "#5da39d": "#1f6b65", "#6f89c4": "#325299",
    "#9b7eb8": "#65458a", "#79a37b": "#356938", "#b5a05e": "#735e19", "#b87896": "#8a3d62",
});
/** "#rrggbb" + alpha -> "rgba(r,g,b,a)" (canvas cannot take var(), and the
 *  pack colour is needed at several strengths). */
export function rgba(hex, a) {
    const h = String(hex).replace("#", "");
    const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

export function stripeFor(folder) {
    const f = String(folder || "").toLowerCase().replace(/[^a-z0-9]/g, "");
    for (const [key, hue] of PACK_STRIPE) if (f.includes(key)) return hue;
    return "#78aec9";   // CustomNodePacks and anything unknown: Glacier Steel, the house accent
}

// The wolf mark (same outline as c2c_ui/wolf.js), in a 48-unit box.
const WOLF_D = "M 11 3 L 17.5 14.5 L 24 12.5 L 30.5 14.5 L 37 3 L 41 19 L 45 27 L 37 31 "
    + "L 29.5 38.5 L 24 45.5 L 18.5 38.5 L 11 31 L 3 27 L 7 19 Z "
    + "M 14.5 23.5 L 21 25 L 20 27.5 L 14 25.8 Z M 33.5 23.5 L 27 25 L 28 27.5 L 34 25.8 Z "
    + "M 21.6 38.2 L 26.4 38.2 L 24 41 Z M 11.2 8.5 L 14.6 14.8 L 9.6 17.2 Z M 36.8 8.5 L 33.4 14.8 L 38.4 17.2 Z";
let _wolf = null;
const wolfPath = () => (_wolf ||= (typeof Path2D === "function" ? new Path2D(WOLF_D) : null));


/** "/extensions/<folder>/..." -> "<folder>", or null. */
export function servedFolder(url) {
    try {
        const path = decodeURIComponent(new URL(url).pathname);
        const m = path.match(/\/extensions\/([^/]+)\//);
        return m ? m[1] : null;
    } catch {
        return null;
    }
}

/** Lower-case letters and digits only: "ComfyUI-NukeMaxNodes" and
 *  "comfyui-nukemax-nodes" both become "comfyuinukemaxnodes". */
const _norm = (s) => String(s).toLowerCase().replace(/[^a-z0-9]/g, "");

/** Does this node definition belong to the pack in `folder`?
 *
 *  ComfyUI serves a pack's web files under its module folder name - UNLESS the
 *  pack's pyproject.toml declares [tool.comfy] web, in which case they are
 *  served under the pyproject project name (nodes.py, load_custom_node). The
 *  two differ only in case and separators, so they are compared normalised;
 *  otherwise one added pyproject line would silently switch the branding off.
 */
export function belongsTo(nodeData, folder) {
    if (!folder) return false;
    const mod = nodeData?.python_module;
    if (typeof mod !== "string" || !mod) return false;
    // "custom_nodes.<folder>" - the folder itself may contain dots
    const own = mod.startsWith("custom_nodes.") ? mod.slice("custom_nodes.".length) : mod;
    return _norm(own) === _norm(folder);
}

/** Grow `node` to its own minimum size; never shrink either side. Only for a
 *  node carrying a DOM widget (a plain textarea does not count). */
export function fitToContent(node) {
    try {
        if (!node || node.isVirtualNode) return false;
        const dom = (node.widgets || []).some((w) => w?.element && w.element.tagName !== "TEXTAREA");
        if (!dom) return false;
        const min = node.computeSize?.();
        if (!min || !node.size) return false;
        const w = Math.max(node.size[0], min[0]);
        // Nodes 2.0 sizes a node to its content and treats size[1] as a MINIMUM;
        // computeSize also counts widgets it hides (advanced inputs), so growing
        // the height there only added dead space (ReLight2D: 1226 -> 1470 px).
        const vue = !!globalThis.LiteGraph?.vueNodesMode;
        const h = vue ? node.size[1] : Math.max(node.size[1], min[1]);
        if (w === node.size[0] && h === node.size[1]) return false;
        node.setSize([w, h]);
        node.setDirtyCanvas?.(true, true);
        return true;
    } catch {
        return false;   // sizing is cosmetic; it must never break the node
    }
}

/** True when the node wears our palette (never set, set by us, or a legacy
 *  colour of ours). A colour the user picked is theirs: drawn flat. */
export function wearsBrand(node) {
    const c = node?.color;
    return !c || c === BRAND.title || LEGACY_TITLES.has(c);
}

/** Bring a node saved with an older C2C palette up to the current one. */
export function upgradeLegacy(node) {
    if (!node) return;
    if (LEGACY_TITLES.has(node.color)) node.color = BRAND.title;
    if (LEGACY_BODIES.has(node.bgcolor)) node.bgcolor = BRAND.body;
}

const _radius = () => (window.LiteGraph?.ROUND_RADIUS ?? 8);

const _gradCache = new WeakMap();

function _isLiteChrome() {
    try { return globalThis.__c2cRuntime?.tier?.() === "lite"; } catch (_) { return false; }
}

function _gradKey(node, th, scale, stripe) {
    const w = node.size[0];
    const h = node.flags?.collapsed ? 0 : node.size[1];
    const collapsed = node.flags?.collapsed ? 1 : 0;
    const scaleBucket = (scale ?? 1) < 0.5 ? 0 : 1;
    const lite = _isLiteChrome() ? 1 : 0;
    return `${w}|${h}|${th}|${stripe}|${scaleBucket}|${collapsed}|${lite}`;
}

function _cacheGrads(node, key, build) {
    let rec = _gradCache.get(node);
    if (rec && rec.key === key) return rec.grads;
    const grads = build();
    _gradCache.set(node, { key, grads });
    return grads;
}

function _titlePath(ctx, w, th, collapsed, shape) {
    const L = window.LiteGraph || {};
    ctx.beginPath();
    if (shape === L.BOX_SHAPE || typeof ctx.roundRect !== "function") { ctx.rect(0, -th, w, th); return; }
    const r = _radius();
    ctx.roundRect(0, -th, w, th, collapsed ? r : [r, r, 0, 0]);
}

/** Title bar: a dark gradient, a faint top highlight, and a hairline under it
 *  that fades out from the pack's colour. A colour the user set is drawn flat. */
function drawTitleBar(node, ctx, th, size, scale, stripe) {
    const w = size?.[0] ?? node.size[0];
    const collapsed = !!node.flags?.collapsed;
    const shape = node.renderingShape ?? node.shape;
    const ours = wearsBrand(node);
    const lite = _isLiteChrome();
    const s = scale ?? 1;
    const key = _gradKey(node, th, s, stripe) + ":title";
    const grads = (!lite && ours && s >= 0.5)
        ? _cacheGrads(node, key, () => {
            const title = ctx.createLinearGradient(0, -th, 0, 0);
            title.addColorStop(0, BRAND.titleTop);
            title.addColorStop(1, BRAND.titleBottom);
            return { title };
        })
        : null;
    ctx.save();
    _titlePath(ctx, w, th, collapsed, shape);
    if (ours && s >= 0.5) {
        ctx.fillStyle = lite ? BRAND.titleTop : grads.title;
    } else {
        ctx.fillStyle = node.renderingColor || node.color || BRAND.title;
    }
    ctx.fill();
    if (s >= 0.5) {
        ctx.clip();
        // NIGHTCALL restraint (PALETTE_NIGHTCALL 4.3 / 7.6): a flat header - no aurora wash - and one crisp 1px
        // divider in the pack's colour under it.
        ctx.fillStyle = "rgba(255,255,255,0.05)";
        ctx.fillRect(0, -th, w, 1);
        if (!collapsed && ours) {
            ctx.fillStyle = rgba(stripe, 0.55);
            ctx.fillRect(0, -1, w, 1);
        }
    }
    ctx.restore();
}

/** The collapse box, drawn as the wolf badge: a dark disc, a ring and the
 *  wolf in the pack's colour. LiteGraph's hit-testing is unchanged. */
function drawTitleBox(node, ctx, th, scale, stripe) {
    const path = wolfPath();
    const cx = th * 0.5, cy = -th * 0.5;
    ctx.save();
    if (!path || scale < 0.5) {
        ctx.fillStyle = stripe;
        ctx.beginPath();
        ctx.arc(cx, cy, 4, 0, Math.PI * 2);
        ctx.fill();
    } else {
        ctx.fillStyle = BRAND.badge;
        ctx.beginPath();
        ctx.arc(cx, cy, 10, 0, Math.PI * 2);
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = rgba(stripe, 0.9);
        ctx.stroke();
        const s = 13 / 48;
        ctx.translate(cx - 6.5, cy - 6.5);
        ctx.scale(s, s);
        ctx.fillStyle = stripe;
        ctx.fill(path, "evenodd");
    }
    ctx.restore();
}

function _nodePath(ctx, node, th, inset = 0) {
    const w = node.size[0], h = (node.flags?.collapsed ? 0 : node.size[1]) + th;
    const L = window.LiteGraph || {};
    const cw = node.flags?.collapsed ? (node._collapsed_width || w) : w;
    ctx.beginPath();
    if ((node.renderingShape ?? node.shape) === L.BOX_SHAPE || typeof ctx.roundRect !== "function") {
        ctx.rect(-inset, -th - inset, cw + 2 * inset, h + 2 * inset);
    } else {
        ctx.roundRect(-inset, -th - inset, cw + 2 * inset, h + 2 * inset, _radius() + inset);
    }
}

/** Under the widgets: nothing any more. NIGHTCALL (owner-approved 2026-11-28) dropped the outer glow, the body bloom
 *  and the left spine; the node is told apart by its 1px rim and wolf disc. Kept as a function so callers and other
 *  packs' copies keep working. */
function drawUnderlay(node, ctx, th, scale, stripe) {   // eslint-disable-line no-unused-vars
}

/** On top: a crisp, flat 1px rim in the pack's colour at 0.75 (>= 3.3:1 on the body for every pack, WCAG 1.4.11);
 *  LiteGraph draws its own selection outline over it. */
function drawOutline(node, ctx, th, scale, stripe = "#78aec9") {
    if (scale < 0.5 || !wearsBrand(node)) return;
    ctx.save();
    ctx.strokeStyle = rgba(stripe, node.is_selected ? 0.95 : 0.75);
    ctx.lineWidth = 1;
    _nodePath(ctx, node, th, -0.5);
    ctx.stroke();
    ctx.restore();
}

const WIDGET_KEYS = ["WIDGET_BGCOLOR", "WIDGET_OUTLINE_COLOR", "WIDGET_TEXT_COLOR", "WIDGET_SECONDARY_TEXT_COLOR"];
const FIELD_RADIUS = 5;

/** Run `draw` with LiteGraph's widget colours set to ours, then put them back.
 *  Widgets read these constants at draw time, so this themes our nodes'
 *  widgets and nothing else. Synchronous: no other node draws in between.
 *
 *  With `stripe` the field rim is tinted in the pack's colour; with `ctx` the
 *  field is drawn as a field, not LiteGraph's pill. A widget's shape is
 *  `roundRect(x, y, w, h, [h / 2])` (BaseWidget.drawWidgetShape) - only that
 *  exact call is squared off, for the length of this draw. */
export function withNightWidgets(draw, stripe, ctx) {
    const L = window.LiteGraph;
    if (!L) return draw();
    const saved = WIDGET_KEYS.map((k) => L[k]);
    L.WIDGET_BGCOLOR = BRAND.widgetBg;
    L.WIDGET_OUTLINE_COLOR = BRAND.widgetOutline;   // neutral fields; the pack colour is for focus / edge only
    L.WIDGET_TEXT_COLOR = BRAND.widgetText;
    L.WIDGET_SECONDARY_TEXT_COLOR = BRAND.widgetText2;
    const rr = typeof ctx?.roundRect === "function" ? ctx.roundRect : null;
    const own = rr && Object.prototype.hasOwnProperty.call(ctx, "roundRect");
    if (rr) {
        ctx.roundRect = function (x, y, w, h, r) {
            if (Array.isArray(r) && r.length === 1 && h <= 40 && Math.abs(r[0] - h / 2) < 0.01) r = [FIELD_RADIUS];
            return rr.call(this, x, y, w, h, r);
        };
    }
    try {
        return draw();
    } finally {
        WIDGET_KEYS.forEach((k, i) => { L[k] = saved[i]; });
        if (rr) { if (own) ctx.roundRect = rr; else delete ctx.roundRect; }
    }
}

function enabled() {
    try {
        const v = app.ui?.settings?.getSettingValue?.("c2c.brand.nodeColors", true);
        return v !== false;
    } catch {
        return true;
    }
}

// ── Nodes 2.0 (Vue renderer) ────────────────────────────────────────────────
// There each node is a DOM element: `.lg-node[data-node-id]` holding a
// `[data-testid="node-inner-wrapper"]` whose inline style already carries the
// node's own colours (background = node.color, --component-node-background =
// node.bgcolor), so the night palette arrives on its own. What CSS adds is the
// part that makes our nodes stand out, the same language as the classic
// canvas: a 1px rim in the pack's colour, a flat header with a 1px divider and the wolf badge (NIGHTCALL); a
// crisp divider and neutral fields; the pack colour is for the edge, focus and active controls only.
//
// Vue rewrites `class` on every render, so a class we added would vanish; it
// never binds `data-c2c-pack`, so that attribute survives. Every layer is
// translucent on purpose: a colour the user picked still shows through.
// Nothing is inserted into Vue's markup; the spine and badge are pseudo-elements.
const PACK_KEYS = ["nukemax", "ltxenhanced", "minimax", "wannodeexperiments", "wananimatepreprocess",
    "wananimalpreprocess", "glm", "omniscale"];
export function packKey(folder) {
    const f = String(folder || "").toLowerCase().replace(/[^a-z0-9]/g, "");
    if (f.includes("nukenode")) return "nukemax";
    if (f.includes("ltxe")) return "ltxenhanced";
    if (f.includes("magnific")) return "omniscale";
    return PACK_KEYS.find((k) => f.includes(k)) || "core";
}

const VUE_STRIPES = [["core", "#78aec9"], ["nukemax", "#c28352"], ["ltxenhanced", "#c9666c"],
    ["minimax", "#5da39d"], ["wannodeexperiments", "#6f89c4"], ["wananimatepreprocess", "#9b7eb8"],
    ["wananimalpreprocess", "#79a37b"], ["glm", "#b5a05e"], ["omniscale", "#b87896"]];

export function vueChromeCss() {
    const wolfFor = (hex) => "data:image/svg+xml," + encodeURIComponent(
        `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"><path fill="${hex}" `
        + `fill-rule="evenodd" d="${WOLF_D}"/></svg>`);
    const N = ".lg-node[data-c2c-pack]";
    const W = `${N} > [data-testid="node-inner-wrapper"]`;
    const S = "var(--c2c-node-stripe)";
    const mix = (pct) => `color-mix(in srgb, ${S} ${pct}%, transparent)`;
    return `
${VUE_STRIPES.map(([k, c]) => `.lg-node[data-c2c-pack="${k}"] { --c2c-node-stripe: ${c}; --c2c-node-wolf: url("${wolfFor(c)}"); }`).join("\n")}
/* NIGHTCALL (owner-approved 2026-11-28): dark body, one crisp 1px rim in the pack's colour, a flat header with a 1px
   divider, the wolf disc - no glow, no spine, no aurora, no bloom. The node's own colours still come from Vue's
   inline style (node.color / node.bgcolor), so a colour the user picks still shows. */
${N} {
  /* core paints number-slider fills, toggles and progress in azure; ours take the pack's colour instead */
  --primary-background: ${S};
  --primary-background-hover: ${S};
  --node-component-header-icon: ${BRAND.ink2};
}
${W} {
  border-radius: 10px !important;
  box-shadow:
    0 0 0 1px ${mix(75)},
    0 10px 24px -10px rgba(0,0,0,0.65);
}
${N}.selected > [data-testid="node-inner-wrapper"], ${N}[data-selected] > [data-testid="node-inner-wrapper"] {
  box-shadow: 0 0 0 1px ${mix(95)}, 0 10px 24px -10px rgba(0,0,0,0.65);
}
${N} .lg-node-header {
  position: relative;
  color: ${BRAND.ink};
  font-weight: 600;
  letter-spacing: 0.01em;
  padding-right: 36px !important;
  border-bottom: 1px solid ${mix(55)};
}
${N} [data-testid="decrement"], ${N} [data-testid="increment"] {
  color: ${BRAND.widgetText2};
}
${N} .lg-node-header::after {
  content: "";
  position: absolute; right: 9px; top: 50%;
  width: 22px; height: 22px; transform: translateY(-50%);
  border-radius: 50%;
  background: ${BRAND.badge} var(--c2c-node-wolf) center / 14px no-repeat;
  box-shadow: 0 0 0 1px ${mix(80)};
  pointer-events: none;
}
${N} input:focus-visible, ${N} textarea:focus-visible, ${N} select:focus-visible {
  outline: 1px solid ${S};
  outline-offset: 0;
}
/* in-node C2C controls take this pack's accent; standalone panels keep the house accent */
${N} .c2c-ui { --cu-accent: ${S}; }
.dark-theme ${N} {
  --component-node-widget-background: ${BRAND.widgetBg};
  --component-node-widget-background-hovered: #10141d;
  --component-node-widget-background-selected: #1a1f2b;
  --component-node-border: ${BRAND.widgetOutline};
  --component-node-foreground: ${BRAND.widgetText};
  --component-node-foreground-secondary: ${BRAND.widgetText2};
  --node-component-slot-text: ${BRAND.ink2};
}
`;
}

/** The pack a Vue node element belongs to, or "" when it is not ours. The
 *  graph on screen, not the root: inside a subgraph the ids are the subgraph's
 *  own and collide with the root's. */
function packOfElement(el) {
    const g = app.canvas?.graph ?? app.graph;
    const raw = el.dataset.nodeId;
    const n = g?.getNodeById?.(raw) ?? g?.getNodeById?.(Number(raw));
    return n?.__c2cPack || "";
}

function tagVueNode(el) {
    const pack = packOfElement(el);
    if (pack) { if (el.dataset.c2cPack !== pack) el.dataset.c2cPack = pack; }
    else if (el.dataset.c2cPack) delete el.dataset.c2cPack;
}

/** Re-check every Vue node element. Vue keys node elements by id, so loading
 *  another workflow can hand node 7's element to a different node 7. */
export function retagVueNodes() {
    if (typeof document === "undefined") return;
    const run = () => document.querySelectorAll(".lg-node[data-node-id]").forEach(tagVueNode);
    run();
    requestAnimationFrame?.(run);
}

/** Re-check the element of one node, once Vue has drawn it. */
export function retagVueNode(node) {
    if (typeof document === "undefined" || node?.id == null) return;
    const run = () => {
        const el = document.querySelector(`.lg-node[data-node-id="${node.id}"]`);
        if (el) tagVueNode(el);
    };
    requestAnimationFrame?.(run);
}

/** One stylesheet and one observer for the whole page, however many packs
 *  carry a copy of this file. The observer does nothing while the classic
 *  renderer is active. It re-checks a node element when one is added AND when
 *  the inside of one changes: Vue keys node elements by id, so a new node that
 *  takes a freed id (clear + add in one tick, a subgraph's own id 1) keeps the
 *  old element and only its contents are re-rendered. */
export function installVueChrome() {
    if (typeof document === "undefined" || window.__C2C_VUE_CHROME__) return;
    window.__C2C_VUE_CHROME__ = true;
    const st = document.createElement("style");
    st.id = "c2c-node-chrome";
    st.textContent = vueChromeCss();
    document.head.appendChild(st);
    if (typeof MutationObserver !== "function") return;
    const SEL = ".lg-node[data-node-id]";
    const obs = new MutationObserver((records) => {
        if (!window.LiteGraph?.vueNodesMode) return;
        const seen = new Set();
        const tag = (el) => { if (el && !seen.has(el)) { seen.add(el); tagVueNode(el); } };
        for (const r of records) {
            const host = r.target?.closest?.(SEL);
            if (host) tag(host);
            for (const n of r.addedNodes) {
                if (n.nodeType !== 1) continue;
                if (n.matches(SEL)) tag(n);
                else if (n.firstElementChild) n.querySelectorAll(SEL).forEach(tag);
            }
        }
    });
    obs.observe(document.body, { childList: true, subtree: true });
    retagVueNodes();
}

const FOLDER = servedFolder(import.meta.url);

// The setting is registered once, by whichever pack loads first; every copy
// reads it. Same guard pattern as the theme's settings.
if (!window.__C2C_BRAND_REG__) { window.__C2C_BRAND_REG__ = true;
    app.registerExtension({
        name: "C2C.BrandSettings",
        settings: [
            {
                id: "c2c.brand.nodeColors",
                name: "C2C → Night node style",
                tooltip: "Code2Collapse nodes get their own look: night title bar with the wolf mark, "
                    + "a stripe in each pack's colour, an outline and night widgets. "
                    + "A colour you set yourself always wins. Turning it off takes effect after a reload.",
                type: "boolean",
                defaultValue: true,
            },
        ],
    });
}

const STRIPE = stripeFor(FOLDER);

/** Install the chrome on one of our node types. Every hook defers to what the
 *  node already had. */
function installChrome(nodeType) {
    const P = nodeType.prototype;
    if (P.__c2cChrome) return;
    P.__c2cChrome = true;
    P.__c2cPack = packKey(FOLDER);

    // Semibold title. LiteGraph reads the title font from this getter both to
    // draw the title and to measure it in computeSize, so the two stay in step.
    Object.defineProperty(P, "titleFontStyle", {
        configurable: true,
        get() {
            const L = window.LiteGraph || {};
            return `600 ${L.NODE_TEXT_SIZE ?? 14}px ${L.NODE_FONT ?? "Arial"}`;
        },
    });

    const ownTitleBar = P.onDrawTitleBar;
    P.onDrawTitleBar = function (ctx, th, size, scale) {
        if (ownTitleBar) return ownTitleBar.apply(this, arguments);
        try { drawTitleBar(this, ctx, th, size, scale ?? 1, STRIPE); }
        catch { /* cosmetic — do not abort the frame */ }
    };
    const ownTitleBox = P.onDrawTitleBox;
    P.onDrawTitleBox = function (ctx, th, size, scale) {
        if (ownTitleBox) return ownTitleBox.apply(this, arguments);
        try { drawTitleBox(this, ctx, th, scale ?? 1, STRIPE); }
        catch { /* cosmetic */ }
    };
    const ownFg = P.onDrawForeground;
    P.onDrawForeground = function (ctx) {
        const r = ownFg?.apply(this, arguments);
        try {
            const th = window.LiteGraph?.NODE_TITLE_HEIGHT ?? 30;
            drawOutline(this, ctx, th, window.comfyAPI?.app?.app?.canvas?.ds?.scale ?? 1, STRIPE);
        } catch { /* cosmetic */ }
        return r;
    };
    const ownBg = P.onDrawBackground;
    P.onDrawBackground = function (ctx) {
        try {
            const th = window.LiteGraph?.NODE_TITLE_HEIGHT ?? 30;
            drawUnderlay(this, ctx, th, window.comfyAPI?.app?.app?.canvas?.ds?.scale ?? 1, STRIPE);
        } catch { /* cosmetic */ }
        return ownBg?.apply(this, arguments);
    };
    const drawWidgets = P.drawWidgets;
    if (typeof drawWidgets === "function") {
        P.drawWidgets = function (...a) {
            if (!wearsBrand(this)) return drawWidgets.apply(this, a);
            return withNightWidgets(() => drawWidgets.apply(this, a), STRIPE, a[0]);
        };
    }
}

app.registerExtension({
    name: `C2C.Brand.${FOLDER || "unknown"}`,
    setup() { if (enabled()) installVueChrome(); },
    afterConfigureGraph() { if (enabled()) retagVueNodes(); },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!belongsTo(nodeData, FOLDER)) return;
        if (!nodeType.title_text_color) nodeType.title_text_color = BRAND.ink;
        if (enabled()) installChrome(nodeType);
        const orig = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = orig?.apply(this, arguments);
            if (enabled()) {
                if (!this.color) this.color = BRAND.title;
                if (!this.bgcolor) this.bgcolor = BRAND.body;
            }
            // after every other onNodeCreated (kits add their DOM widgets
            // there) and after a loaded workflow restores the saved size and
            // colours (configure runs after onNodeCreated)
            const node = this;
            requestAnimationFrame(() => {
                if (enabled()) upgradeLegacy(node);
                fitToContent(node);
                if (enabled() && globalThis.LiteGraph?.vueNodesMode) retagVueNode(node);
            });
            return r;
        };
    },
});
