// C2C Node Identity - title visibility in BOTH renderers (design docs/research/node_identity_design.md s2).
//
// Classic LiteGraph canvas: a hidden node gets an OWN-PROPERTY onDrawTitleText hook (fe 1.53.10
// LGraphNode.drawTitleText calls it INSTEAD of drawing the text and returns). Title bar, collapse box, badges, title
// buttons, selection outline and progress bar are separate draw calls and stay. No prototype is touched; showing the
// title again deletes the own property (or restores an own hook that was there before).
// Nodes 2.0 (Vue): one <style> element with per-node-id rules for the graph on screen; `visibility:hidden` keeps
// the header box, so header height / node size never change. Badges, pin and Muted/Bypassed are siblings of the
// title div in NodeHeader.vue and are never matched.
// Refreshed only on events (no polling, no per-frame walks).
// License: Apache-2.0
import { walk, identityOf, defaultTitleOf } from "./adapter.js";

const STYLE_ID = "c2c-node-identity-style";
const MARK = "__c2cNodeIdentity";

const own = (o, k) => Object.prototype.hasOwnProperty.call(o, k);

export function overrideOf(node) {
    const t = identityOf(node)?.title;
    return t === "show" || t === "hide" ? t : null;
}

/** A CSS string literal. Node ids and types come from workflow files (untrusted): backslash, quote and line
 * breaks are escaped so the value can never close the string and inject a rule. */
export function cssString(s) {
    return `"${String(s).replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/[\r\n\f]+/g, " ")}"`;
}

/** Effective hidden state of one node for a state {enabled, hide}. */
export function isHidden(node, state) {
    if (!state.enabled) return false;
    const o = overrideOf(node);
    return o ? o === "hide" : !!state.hide;
}

/** Nodes 2.0 rules for the nodes of the graph on screen (pure: the Node tests run it). */
export function vueCss(nodes, state) {
    const T = '[data-testid="node-title"]';
    const rules = [];
    for (const node of nodes) {
        if (!isHidden(node, state)) continue;
        const sel = `.lg-node[data-node-id=${cssString(node.id)}]`;
        rules.push(`${sel} ${T} > .truncate:not(:has(input)){visibility:hidden}`);
        if (state.hover) {
            rules.push(`${sel} .lg-node-header:hover ${T} > .truncate{visibility:visible;opacity:.8}`);
            const type = String(node.comfyClass || node.type || "");
            rules.push(`${sel} .lg-node-header:hover ${T}::after{content:${cssString(type)};opacity:.6;`
                + "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:45%;font-size:.92em}");
        }
    }
    return rules.join("\n");
}

function truncate(ctx, text, width) {
    if (width <= 0) return "";
    if (ctx.measureText(text).width <= width) return text;
    let lo = 0;
    let hi = text.length;
    while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        if (ctx.measureText(text.slice(0, mid) + "…").width <= width) lo = mid;
        else hi = mid - 1;
    }
    return lo > 0 ? text.slice(0, lo) + "…" : "";
}

export function createTitleController(app) {
    const state = { enabled: true, hide: false, hover: true };

    const hiddenNode = (node) => isHidden(node, state);

    function hook(ctx, titleHeight, size, _scale, font, selected) {
        const LG = globalThis.LiteGraph || {};
        const y = (LG.NODE_TITLE_TEXT_Y ?? 20) - titleHeight;
        const hovering = state.hover && !!this.mouseOver;
        if (!hovering && !this.pinned) return;
        ctx.save();
        try {
            ctx.font = font || this.titleFontStyle || ctx.font;
            ctx.textAlign = "left";
            ctx.fillStyle = selected ? (LG.NODE_SELECTED_TITLE_COLOR || "#FFF")
                : (this.constructor?.title_text_color || app.canvas?.node_title_color || LG.NODE_TITLE_COLOR || "#999");
            const width = (size?.[0] ?? this.size?.[0] ?? 100) - titleHeight * 2;
            let text = "";
            if (hovering) {
                const type = String(this.comfyClass || this.type || "");
                const title = String(this.title ?? "");
                text = title && title !== type ? `${title}  ·  ${type}` : type;
                ctx.globalAlpha *= 0.8;
            }
            if (this.pinned) text = text ? `${text}📌` : "📌";          // the status glyph the default title shows
            ctx.fillText(truncate(ctx, text, Math.max(width, 24)), titleHeight, y);
        } finally {
            ctx.restore();
        }
    }

    function install(node) {
        const cur = own(node, "onDrawTitleText") ? node.onDrawTitleText : undefined;
        if (cur && cur[MARK]) return;
        const fn = function (...args) { return hook.apply(this, args); };
        fn[MARK] = true;
        fn.prevOwn = cur;
        node.onDrawTitleText = fn;
    }

    function uninstall(node) {
        if (!own(node, "onDrawTitleText")) return;
        const fn = node.onDrawTitleText;
        if (!fn || !fn[MARK]) return;
        if (fn.prevOwn !== undefined) node.onDrawTitleText = fn.prevOwn;
        else delete node.onDrawTitleText;
    }

    function styleEl() {
        let el = document.getElementById(STYLE_ID);
        if (!el) {
            el = document.createElement("style");
            el.id = STYLE_ID;
            document.head.appendChild(el);
        }
        return el;
    }

    /** Apply the current state to every node (classic hooks) and to the graph on screen (Vue rules). */
    function refresh() {
        const root = app.rootGraph ?? app.graph;
        if (!root) return { hidden: 0 };
        let hidden = 0;
        for (const { node } of walk(root)) {
            if (hiddenNode(node)) { install(node); hidden++; } else uninstall(node);
        }
        const shown = app.canvas?.graph || root;
        try {
            styleEl().textContent = vueCss(shown.nodes || shown._nodes || [], state);
        } catch (_e) { /* no DOM (tests) */ }
        app.canvas?.setDirty?.(true, true);
        return { hidden };
    }

    function set(patch) {
        Object.assign(state, patch);
        return refresh();
    }

    return { state, refresh, set, hiddenNode, install, uninstall, defaultTitleOf };
}
