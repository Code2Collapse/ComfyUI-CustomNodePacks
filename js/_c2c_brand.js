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
// EDGE carries the node: a rim, an outer glow, a spine and a title aurora in the
// pack's colour, plus the wolf badge. `title` is the flat equivalent of the
// title gradient, used where only one colour can be given (collapsed nodes,
// menus, the Nodes 2.0 wrapper).
export const BRAND = Object.freeze({
    // "Obsidian night": every shade is a step of the night ramp in
    // _c2c_theme.js (bg3 #07081a .. surface2 #3b3a68, the swatches the user
    // gave), with the darkness put back. The body is DARKER than ComfyUI's
    // canvas; what separates a C2C node from it is the rim, glow and aurora in
    // the pack's colour, not a bright fill (the old #282a56 read as "full purple").
    title: "#12132f",           // night bg: collapsed nodes, menus, Nodes 2.0 header
    body: "#0c0d23",            // night bg2
    ink: "#f3f1ff",
    ink2: "#bab7db",            // night sub
    titleTop: "#1d1e45",        // night surface0
    titleBottom: "#0f1030",     // the darkest swatch of the user's ramp
    badge: "#07081a",           // night bg3: wolf badge disc
    widgetBg: "#07081a",        // night bg3: fields sit below the body
    widgetOutline: "#2a2a57",   // night surface1 (a pack-tinted rim replaces it on our nodes)
    widgetText: "#e8e6f7",      // night fg
    widgetText2: "#8280ba",     // night overlay2
});

// Colours this file handed out before. A workflow saved since then stores
// them on each node; they are ours, not a choice the user made, so they are
// upgraded rather than preserved.
const LEGACY_TITLES = new Set(["#26275a", "#33357c", "#11122b"]);
const LEGACY_BODIES = new Set(["#17182f", "#282a56", "#0b0c1d"]);

// One accent per pack: a thin line across the top of the title bar.
const PACK_STRIPE = [
    ["nukemax", "#f1aa7b"], ["nukenode", "#f1aa7b"],
    ["minimax", "#76dccb"],
    ["wannodeexperiments", "#7fd4f2"],
    ["wananimatepreprocess", "#eba2de"],
    ["wananimalpreprocess", "#f3d288"],
    ["glm", "#8ee09d"],
];
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
    return "#b494ff";   // CustomNodePacks and anything unknown: the house violet
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
            const aurora = ctx.createLinearGradient(0, 0, Math.max(w * 0.7, 120), 0);
            aurora.addColorStop(0, rgba(stripe, 0.34));
            aurora.addColorStop(1, rgba(stripe, 0));
            const hairline = ctx.createLinearGradient(0, 0, w * 0.8, 0);
            hairline.addColorStop(0, rgba(stripe, 0.95));
            hairline.addColorStop(1, rgba(stripe, 0));
            return { title, aurora, hairline };
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
        if (ours) {
            if (lite) {
                ctx.fillStyle = rgba(stripe, 0.34);
                ctx.fillRect(0, -th, Math.min(w * 0.45, 120), th);
            } else {
                ctx.fillStyle = grads.aurora;
                ctx.fillRect(0, -th, w, th);
            }
        }
        ctx.fillStyle = "rgba(255,255,255,0.07)";
        ctx.fillRect(0, -th, w, 1);
        if (!collapsed && ours) {
            if (lite) {
                ctx.fillStyle = rgba(stripe, 0.95);
                ctx.fillRect(0, -1, Math.min(w * 0.35, 80), 1);
            } else {
                ctx.fillStyle = grads.hairline;
                ctx.fillRect(0, -1, w, 1);
            }
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

/** Under the widgets: a soft glow outside the node in the pack's colour and a
 *  glowing spine down the body's left edge. The glow is two widening strokes,
 *  not canvas shadowBlur - blur on every node every frame costs real CPU. */
function drawUnderlay(node, ctx, th, scale, stripe) {
    if (scale < 0.5 || !wearsBrand(node)) return;
    const lite = _isLiteChrome();
    const key = _gradKey(node, th, scale, stripe) + ":under";
    ctx.save();
    ctx.lineWidth = 3;
    ctx.strokeStyle = rgba(stripe, 0.10);
    _nodePath(ctx, node, th, 2.5);
    ctx.stroke();
    ctx.lineWidth = 2;
    ctx.strokeStyle = rgba(stripe, 0.20);
    _nodePath(ctx, node, th, 1);
    ctx.stroke();
    if (!node.flags?.collapsed) {
        _nodePath(ctx, node, th);
        ctx.clip();
        const h = node.size[1];
        const r = Math.max(node.size[0], 200) * 0.85;
        if (lite) {
            ctx.fillStyle = rgba(stripe, 0.10);
            ctx.fillRect(0, 0, Math.min(r, node.size[0]), Math.min(r * 0.5, h));
            ctx.fillStyle = rgba(stripe, 0.95);
            ctx.fillRect(0, 0, 3, h);
        } else {
            const grads = _cacheGrads(node, key, () => {
                const glow = ctx.createRadialGradient(0, 0, 0, 0, 0, r);
                glow.addColorStop(0, rgba(stripe, 0.10));
                glow.addColorStop(1, rgba(stripe, 0));
                const spine = ctx.createLinearGradient(0, 0, 0, h);
                spine.addColorStop(0, rgba(stripe, 0.95));
                spine.addColorStop(1, rgba(stripe, 0.08));
                return { glow, spine };
            });
            ctx.fillStyle = grads.glow;
            ctx.fillRect(0, 0, Math.min(r, node.size[0]), Math.min(r, h));
            ctx.fillStyle = grads.spine;
            ctx.fillRect(0, 0, 3, h);
        }
    }
    ctx.restore();
}

/** On top: a crisp rim, bright at the top and fading down, in the pack's
 *  colour. LiteGraph draws its own selection outline over it. */
function drawOutline(node, ctx, th, scale, stripe = "#b494ff") {
    if (scale < 0.5 || !wearsBrand(node)) return;
    const h = (node.flags?.collapsed ? 0 : node.size[1]) + th;
    const lite = _isLiteChrome();
    const key = _gradKey(node, th, scale, stripe) + ":outline";
    ctx.save();
    if (lite) {
        ctx.strokeStyle = rgba(stripe, 0.85);
    } else {
        const grads = _cacheGrads(node, key, () => {
            const rim = ctx.createLinearGradient(0, -th, 0, h - th);
            rim.addColorStop(0, rgba(stripe, 0.85));
            rim.addColorStop(1, rgba(stripe, 0.28));
            return { rim };
        });
        ctx.strokeStyle = grads.rim;
    }
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
    L.WIDGET_OUTLINE_COLOR = stripe ? rgba(stripe, 0.30) : BRAND.widgetOutline;
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
// canvas: a rim and glow in the pack's colour, the aurora in the header, a
// glowing spine, pack-tinted fields and the wolf badge in the header.
//
// Vue rewrites `class` on every render, so a class we added would vanish; it
// never binds `data-c2c-pack`, so that attribute survives. Every layer is
// translucent on purpose: a colour the user picked still shows through.
// Nothing is inserted into Vue's markup; the spine and badge are pseudo-elements.
const PACK_KEYS = ["nukemax", "minimax", "wannodeexperiments", "wananimatepreprocess",
    "wananimalpreprocess", "glm"];
export function packKey(folder) {
    const f = String(folder || "").toLowerCase().replace(/[^a-z0-9]/g, "");
    if (f.includes("nukenode")) return "nukemax";
    return PACK_KEYS.find((k) => f.includes(k)) || "core";
}

const VUE_STRIPES = [["core", "#b494ff"], ["nukemax", "#f1aa7b"], ["minimax", "#76dccb"],
    ["wannodeexperiments", "#7fd4f2"], ["wananimatepreprocess", "#eba2de"],
    ["wananimalpreprocess", "#f3d288"], ["glm", "#8ee09d"]];

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
/* Obsidian: dark body, rim + glow in the pack's colour, glowing spine, wolf
   badge in the header. The node's own colours still come from Vue's inline
   style (node.color / node.bgcolor), so a colour the user picks still shows;
   everything here is layered on top of it. */
${N} {
  /* core paints number-slider fills, toggles and progress in azure; ours
     take the pack's colour instead */
  --primary-background: ${S};
  --primary-background-hover: ${S};
  --node-component-header-icon: ${BRAND.ink2};
}
${W} {
  border-radius: 10px !important;
  box-shadow:
    0 0 0 1px ${mix(62)},
    0 0 22px -4px ${mix(42)},
    0 18px 36px -12px rgba(0,0,0,0.78);
}
${N}::before {
  content: "";
  position: absolute; left: 0; top: 40px; bottom: 12px; width: 3px; z-index: 3;
  border-radius: 0 3px 3px 0;
  background: linear-gradient(180deg, ${S}, ${mix(8)});
  box-shadow: 0 0 10px ${mix(55)};
  pointer-events: none;
}
${N}[data-collapsed]::before { display: none; }
${N} .lg-node-header {
  position: relative;
  color: ${BRAND.ink};
  font-weight: 600;
  letter-spacing: 0.01em;
  padding-right: 36px !important;
  border-bottom: 1px solid transparent;
  border-image: linear-gradient(90deg, ${S}, transparent 80%) 1;
  /* the aurora: the pack's colour rising from the left edge, over a faint
     top sheen - both translucent, so a colour the user picked shows through */
  background-image:
    linear-gradient(90deg, ${mix(32)}, transparent 70%),
    linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0));
}
${N} [data-testid^="node-body"] {
  background-image: radial-gradient(90% 60% at 0% 0%, ${mix(11)}, transparent 70%);
}
${N} .bg-component-node-widget-background {
  box-shadow: inset 0 0 0 1px ${mix(26)};
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
  box-shadow: 0 0 0 1.5px ${S}, 0 0 12px -2px ${S};
  pointer-events: none;
}
${N} input:focus-visible, ${N} textarea:focus-visible, ${N} select:focus-visible {
  outline: 1px solid ${S};
  outline-offset: 0;
}
.dark-theme ${N} {
  --component-node-widget-background: ${BRAND.widgetBg};
  --component-node-widget-background-hovered: #0e0f2b;
  --component-node-widget-background-selected: #17183f;
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
