/**
 * c2c_ui/node_panel.js — the only way a C2C node mounts a DOM widget.
 *
 * License: Apache-2.0
 */

import { ensureStyles } from "./theme.js";
import { adaptiveCanvasOnly, installResizeFloor, measureRootContent } from "./nodes2.js";

/**
 * Mount a C2C panel inside a ComfyUI node.
 *
 * ComfyUI sizes the DOM element to computedHeight - 4 - (margin ? 2*margin - 4 : 0).
 * With margin=4 and reserved = minHeight + 2*margin, the element gets exactly minHeight.
 *
 * @param {object} node
 * @param {string} name
 * @param {HTMLElement} root
 * @param {{minHeight: number, margin?: number, fill?: boolean}} opts
 * @returns {object}
 */
export function mountPanel(node, name, root, opts) {
    const margin = opts.margin ?? 4;
    const minHeight = opts.minHeight;
    const reserved = minHeight + 2 * margin;

    ensureStyles();
    root.classList.add("c2c-ui");
    if (opts.fill) {
        root.style.flex = "1";
        root.style.minHeight = "0";
    }

    const widget = node.addDOMWidget(name, "div", root, {
        serialize: false,
        margin,
        getMinHeight: () => reserved,
    });

    widget.computeSize = (w) => [w, reserved];
    adaptiveCanvasOnly(widget);

    const uninstallFloor = installResizeFloor(root, measureRootContent);

    let ro = null;
    if (typeof ResizeObserver !== "undefined") {
        ro = new ResizeObserver((entries) => {
            if (typeof widget.onPanelResize === "function") {
                widget.onPanelResize(entries);
            }
        });
        ro.observe(root);
    }
    widget.onPanelResize = null;

    const origRemoved = node.onRemoved;
    node.onRemoved = function (...args) {
        try { ro?.disconnect(); } catch (_e) { /* ignore */ }
        try { uninstallFloor(); } catch (_e) { /* ignore */ }
        return origRemoved?.apply(this, args);
    };

    return widget;
}
