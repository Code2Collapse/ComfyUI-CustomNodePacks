/**
 * _wan_director_ui.js — single owner for WanDirectorC2C layout hooks.
 *
 * compact / timeline / variant_gate extensions share:
 *   - viewport height cap
 *   - isWidgetVisible (Vue + LiteGraph)
 *   - per-node advanced-panel state (WeakMap, not global)
 *   - wdHideWidget / wdShowWidget markers
 */
export const WD_DEFAULT_W = 640;

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

// Nodes 2.0 (Comfy.VueNodes.Enabled) support: the Vue renderer decides row
// visibility from widget.options.hidden and only rebuilds its widget-row
// snapshot when a widget is ADDED. So (1) hide/show mirror the state into
// options.hidden, (2) wdVueNudge adds+removes a throwaway widget to trigger
// the rebuild (debounced per node per frame; never rendered — the splice
// lands before the next-frame snapshot).
function _wdVueActive() {
    try {
        return window.app?.ui?.settings?.getSettingValue?.("Comfy.VueNodes.Enabled") === true;
    } catch (_) { return false; }
}

export function wdVueNudge(node) {
    if (!node?.widgets || !_wdVueActive() || node.__wdVueNudgePending) return;
    // Only nudge once the Vue component for this node exists. Before the
    // initial mount the fresh snapshot reads options.hidden anyway, and a
    // rebuild racing the first mount detaches the node's DOM widgets
    // (timeline/player elements end up orphaned from their rows).
    if (!document.querySelector(`[data-node-id="${node.id}"]`)) return;
    node.__wdVueNudgePending = true;
    requestAnimationFrame(() => requestAnimationFrame(() => {
        node.__wdVueNudgePending = false;
        try {
            const d = node.addWidget("number", "__wd_vue_sync__", 0, () => {}, { serialize: false });
            const i = node.widgets.indexOf(d);
            if (i >= 0) node.widgets.splice(i, 1);
        } catch (_) { /* node may be mid-removal */ }
    }));
}

// First-instance mount race: when the very first WanDirector of a page load
// is created, Vue can lose the DOM-widget registration for widgets added
// while the node component is mid-mount — the row renders empty and the
// element stays detached forever (a nudge won't fix it; only re-registering
// does). Called from the timeline's liveness interval as a cheap self-heal.
export function wdEnsureDomWidgetsAttached(node) {
    if (!node?.widgets || !_wdVueActive() || node.graph == null) return;
    if (node.__wdReattachBusy) return;
    if (!document.querySelector(`[data-node-id="${node.id}"]`)) return;
    const broken = node.widgets.filter((w) => w?.element && !w.element.isConnected
        && !(w.options?.hidden || w.type === "hidden" || w.hidden));
    if (!broken.length) return;
    node.__wdReattachBusy = true;
    // Two-phase on purpose: remove first, let Vue flush the removal, THEN
    // re-add. Splice+add in one tick keeps the same widget identity in the
    // snapshot diff, so Vue skips re-registration and the element stays
    // detached (verified live). The pause between phases is what re-keys it.
    const specs = broken.map((w) => ({
        name: w.name, type: String(w.type || "div"),
        el: w.element, opts: w.options || {}, cs: w.computeSize,
    }));
    for (const w of broken) {
        const i = node.widgets.indexOf(w);
        if (i >= 0) node.widgets.splice(i, 1);
    }
    setTimeout(() => {
        try {
            for (const s of specs) {
                const w2 = node.addDOMWidget(s.name, s.type, s.el, s.opts);
                if (s.cs) w2.computeSize = s.cs;
            }
        } catch (_) { /* retried on the next liveness tick */ }
        node.__wdReattachBusy = false;
    }, 150);
}

// ── Edit-proxy client (pairs with GET /wne/media_proxy) ─────────────
// The browser can only PLAY web codecs; ProRes/EXR/MXF/… need the server
// to transcode a frame-accurate H.264 proxy once (Resolve/Premiere
// pattern). wdEnsureProxy() returns a playable URL: the original when the
// container is web-safe, else the proxy (polling while it builds).

export function wdViewUrl(file, type = "input") {
    const parts = String(file || "").split("/");
    const name = parts.pop();
    const sub = parts.join("/");
    return `/view?filename=${encodeURIComponent(name)}&subfolder=${encodeURIComponent(sub)}&type=${type}`;
}

export function wdIsWebSafe(file) {
    return /\.(mp4|webm|m4v|ogv)$/i.test(String(file || ""));
}

export async function wdEnsureProxy(file, { height = 720, onProgress, signal } = {}) {
    if (!file) return null;
    if (wdIsWebSafe(file)) return wdViewUrl(file);
    for (let i = 0; i < 400; i++) {            // ~10 min ceiling
        if (signal?.aborted) return null;
        let r = null;
        try {
            r = await (await fetch(
                `/wne/media_proxy?file=${encodeURIComponent(file)}&height=${height}`)).json();
        } catch (_) { return null; }
        if (!r || r.ok === false || r.status === "error") return null;
        if (r.status === "ready" && r.url) return r.url;
        try { onProgress?.(r.progress ?? 0); } catch (_) {}
        await new Promise((res) => setTimeout(res, 1500));
    }
    return null;
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
    w.options = w.options || {};
    w.options.hidden = true;
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
    if (w.options) w.options.hidden = false;
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
