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

// Night palette (see _c2c_theme.js, variant "night"). The title bar sits one
// step above the body so it still reads as a header; both stay well below the
// ground of ComfyUI's own widgets so inputs keep their contrast.
export const BRAND = Object.freeze({
    title: "#26275a",
    body: "#17182f",
    ink: "#e8e6f7",
});


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
        const h = Math.max(node.size[1], min[1]);
        if (w === node.size[0] && h === node.size[1]) return false;
        node.setSize([w, h]);
        node.setDirtyCanvas?.(true, true);
        return true;
    } catch {
        return false;   // sizing is cosmetic; it must never break the node
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

const FOLDER = servedFolder(import.meta.url);

// The setting is registered once, by whichever pack loads first; every copy
// reads it. Same guard pattern as the theme's settings.
if (!window.__C2C_BRAND_REG__) { window.__C2C_BRAND_REG__ = true;
    app.registerExtension({
        name: "C2C.BrandSettings",
        settings: [
            {
                id: "c2c.brand.nodeColors",
                name: "C2C → Brand node colours",
                tooltip: "Give every Code2Collapse node the night title bar and body. "
                    + "A colour you set yourself, or one saved in a workflow, always wins.",
                type: "boolean",
                defaultValue: true,
            },
        ],
    });
}

app.registerExtension({
    name: `C2C.Brand.${FOLDER || "unknown"}`,
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!belongsTo(nodeData, FOLDER)) return;
        if (!nodeType.title_text_color) nodeType.title_text_color = BRAND.ink;
        const orig = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = orig?.apply(this, arguments);
            if (enabled()) {
                if (!this.color) this.color = BRAND.title;
                if (!this.bgcolor) this.bgcolor = BRAND.body;
            }
            // after every other onNodeCreated (kits add their DOM widgets
            // there) and after a loaded workflow restores the saved size
            const node = this;
            requestAnimationFrame(() => fitToContent(node));
            return r;
        };
    },
});
