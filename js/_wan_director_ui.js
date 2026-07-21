/**
 * _wan_director_ui.js — single owner for WanDirectorC2C layout hooks.
 *
 * compact / timeline / variant_gate extensions share:
 *   - viewport height cap
 *   - isWidgetVisible (Vue + LiteGraph)
 *   - per-node advanced-panel state (WeakMap, not global)
 *   - wdHideWidget / wdShowWidget markers
 */
export const WD_DEFAULT_W = 1375;

export function wdMaxH() {
    // 0.62 → 0.88: the 62%-viewport clamp cut the node ~240px shorter than its
    // own widget stack (timeline + player + props), so the DOM spilled out of
    // the node bounds over neighbouring nodes — the reported "UI coming out
    // of the node" bug. The wrapper is also overflow-clipped as a belt-and-
    // braces guard (wan_director_timeline/player).
    return Math.max(520, Math.floor((window.innerHeight || 1080) * 0.88));
}

export function writeWdNodeSize(node, w, h) {
    if (!node) return;
    if (!node.size) node.size = [w, h];
    else {
        node.size[0] = w;
        node.size[1] = h;
    }
}

export function capWdComputeSize(sz) {
    const cap = wdMaxH();
    return [
        Math.max(sz?.[0] || WD_DEFAULT_W, WD_DEFAULT_W),
        Math.min(sz?.[1] || cap, cap),
    ];
}

export function capWdNode(node) {
    if (!node?.size) return;
    const cap = wdMaxH();
    writeWdNodeSize(
        node,
        Math.max(node.size[0] || 0, WD_DEFAULT_W),
        Math.min(node.size[1] || cap, cap),
    );
    node.setDirtyCanvas?.(true, true);
}

export function findWdWidget(node, name) {
    return (node.widgets || []).find((w) => w.name === name);
}

const _advancedOpen = new WeakMap();

export function getWdAdvancedOpen(node) {
    return _advancedOpen.get(node) ?? false;
}

export function setWdAdvancedOpen(node, open) {
    _advancedOpen.set(node, !!open);
}

export function isWdWidgetHidden(w) {
    if (!w) return true;
    return !!(w._wd_ui_hidden || w._wd_compact_hidden || w._wd_hidden);
}

export function wdHideWidget(w) {
    if (!w || w._wd_ui_hidden) return;
    if (w._wd_ui_orig_cs === undefined) w._wd_ui_orig_cs = w.computeSize;
    if (w._wd_ui_orig_type === undefined) w._wd_ui_orig_type = w.type;
    w._wd_ui_hidden = true;
    w.hidden = true;
    w.type = "hidden";
    w.computeSize = () => [0, -4];
    w.draw = () => {};
    const el = w.element;
    if (el) {
        el.hidden = true;
        el.style.display = "none";
        const wrap = el.parentElement;
        if (wrap?.classList?.contains("dom-widget") || wrap?.classList?.contains("lg-node-widget")) {
            wrap.style.display = "none";
        }
    }
}

export function wdShowWidget(w) {
    if (!w || !w._wd_ui_hidden) return;
    w._wd_ui_hidden = false;
    w.hidden = false;
    if (w._wd_ui_orig_type !== undefined) w.type = w._wd_ui_orig_type;
    if (w._wd_ui_orig_cs) w.computeSize = w._wd_ui_orig_cs;
    else delete w.computeSize;
    delete w.draw;
    const el = w.element;
    if (el) {
        el.hidden = false;
        el.style.display = "";
        const wrap = el.parentElement;
        if (wrap?.classList?.contains("dom-widget") || wrap?.classList?.contains("lg-node-widget")) {
            wrap.style.display = "";
        }
    }
}

/** Install prototype hooks once (safe to call from every WanDirector extension). */
export function installWanDirectorPrototype(nodeType) {
    if (nodeType.__c2cWdUiInstalled) return;
    nodeType.__c2cWdUiInstalled = true;

    const _origIsWidgetVisible = nodeType.prototype.isWidgetVisible;
    nodeType.prototype.isWidgetVisible = function (widget) {
        if (isWdWidgetHidden(widget)) return false;
        return _origIsWidgetVisible ? _origIsWidgetVisible.call(this, widget) : true;
    };

    const _origComputeSize = nodeType.prototype.computeSize;
    nodeType.prototype.computeSize = function (outW) {
        const sz = _origComputeSize
            ? _origComputeSize.call(this, outW)
            : [this.size?.[0] || WD_DEFAULT_W, this.size?.[1] || wdMaxH()];
        return capWdComputeSize(sz);
    };

    const _origSetSize = nodeType.prototype.setSize;
    if (typeof _origSetSize === "function") {
        nodeType.prototype.setSize = function (size) {
            const [w, h] = capWdComputeSize(size);
            return _origSetSize.call(this, [w, h]);
        };
    }
}
