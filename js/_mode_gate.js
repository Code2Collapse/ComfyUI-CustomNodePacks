import { reportFailure as __c2cReport } from "./_c2c_report.js";
import { vueSyncNodeWidgets } from "./_widget_visibility.js";

// =====================================================================
// _mode_gate.js — Shared helper for the unified-mode MEC nodes.
//
// Wraps a per-editor installer so it only runs when the node's `mode`
// widget value matches an expected token. Switching the mode at runtime
// hides the editor's host DOM element via display:none and re-runs the
// installer the next time the mode matches.
//
// Usage from a JS extension:
//   import { installModeGated } from "./_mode_gate.js";
//   installModeGated(node, {
//       activeWhen: "edit",            // value of the `mode` widget
//       installerKey: "splineEdit",    // unique key, prevents double-install
//       installer: (n) => myInstallEditor(n),
//       hostFinder: (n) => n._mecEditor?.host || null,
//   });
//
// L7.65 P01 (Image Mask Editor modes), all optional and backward compatible:
//   activeWhen   a mode value, a list of them, or a test (node) => boolean
//   watch        more widgets whose change re-evaluates the gates (default: only `mode`)
//   onActivate   (node, mode) => void, run whenever the gate is evaluated active (after the install)
//   widgetFinder may return a list of widgets (every one is collapsed / restored)
//   keepSize     a restored widget gets back its OWN sizing (computeSize / getHeight ...) instead of a fixed height
// =====================================================================

function _modeValue(node) {
    const w = node?.widgets?.find?.((w) => w && w.name === "mode");
    return w ? String(w.value ?? "") : "";
}

function _isActive(activeWhen, node, mode) {
    if (typeof activeWhen === "function") return !!activeWhen(node);
    if (Array.isArray(activeWhen)) return activeWhen.map(String).includes(mode);
    return mode === String(activeWhen);
}

const _list = (x) => (Array.isArray(x) ? x : (x ? [x] : []));

// Run every registered gate evaluator for a node, but ALWAYS run the one whose
// activeWhen matches the current mode LAST. Several editors can legitimately share
// a single host/DOM widget (e.g. the spline `edit` and `flow_path` editors both
// bind _mecSplineEditHost/_mecSplineEditWidget). If the inactive editors run after
// the active one, their _hideHost collapses the shared widget the active editor
// just restored — leaving the canvas blank in the active mode. Active-last ordering
// guarantees the visible editor's _showHost is the final word on the shared widget.
function _runEvaluators(node) {
    const mode = _modeValue(node);
    const evals = node._mecGateEvaluators || [];
    const ordered = [...evals].sort(
        (a, b) => (_isActive(a.activeWhen, node, mode) ? 1 : 0) - (_isActive(b.activeWhen, node, mode) ? 1 : 0));
    for (const e of ordered) {
        try { e.run(); } catch (err) { __c2cReport("_mode_gate", err); }
    }
    // Nodes 2.0: mirror the batch's hidden state into options.hidden and
    // nudge the Vue widget-row snapshot rebuild.
    try { vueSyncNodeWidgets(node); } catch (err) { __c2cReport("_mode_gate", err); }
}

function _collapseWidget(w) {
    if (!w) return;
    // Remember the widget's real type so _restoreWidget can put it back.
    // Vue nodes (Comfy.VueNodes.Enabled) picks the render component by
    // widget.type — restoring to undefined leaves the widget unmounted there.
    if (w.type !== "hidden") w._mecOrigType = w.type;
    w.type = "hidden";
    w.computeSize = () => [0, -4];
    w.computeLayoutSize = () => ({ minHeight: 0, maxHeight: 0 });
    if (typeof w.getMinHeight === "function") w.getMinHeight = () => 0;
    if (typeof w.getHeight === "function") w.getHeight = () => 0;
    const el = w.element || w.inputEl;
    if (el) {
        el.hidden = true;
        el.style.display = "none";
        const wrap = el.parentElement;
        if (wrap?.classList?.contains("dom-widget")) wrap.style.display = "none";
    }
}

function _restoreWidget(w, height) {
    if (!w) return;
    // Only touch type if we collapsed it; a freshly-installed widget keeps
    // the type addDOMWidget gave it (Vue nodes needs it to mount the element).
    if (w.type === "hidden") w.type = w._mecOrigType;
    // Nodes 2.0: a collapsed widget ends up with `hidden` set (vueSyncNodeWidgets mirrors `hidden` into
    // options.hidden), so without this an editor shown a SECOND time stayed invisible (measured 2026-10-27, L7.54)
    w.hidden = false;
    if (w.options) w.options.hidden = false;
    const h = height || 280;
    w.computeSize = () => [0, h];
    w.computeLayoutSize = undefined;
    if (typeof w.getMinHeight === "function") w.getMinHeight = () => h;
    if (typeof w.getHeight === "function") w.getHeight = () => h;
    const el = w.element || w.inputEl;
    if (el) {
        el.hidden = false;
        el.style.display = "";
        const wrap = el.parentElement;
        if (wrap?.classList?.contains("dom-widget")) wrap.style.display = "";
    }
}

const _SIZERS = ["computeSize", "computeLayoutSize", "getMinHeight", "getHeight"];

function _hideHost(host, widget, keepSize) {
    if (host && host.style) host.style.display = "none";
    for (const w of _list(widget)) {
        // keepSize: the widget sizes itself (e.g. to the image's aspect) - remember its own sizing to put it back
        if (keepSize && w && !w._mecGateSizers) w._mecGateSizers = Object.fromEntries(_SIZERS.map((k) => [k, w[k]]));
        _collapseWidget(w);
    }
}

function _showHost(host, widget, height, keepSize) {
    if (host && host.style) host.style.display = "";
    for (const w of _list(widget)) {
        // keepSize: a widget shown for the first time (never collapsed) keeps the sizing it was made with - restoring
        // a fixed height there gave the video editor's button a 300 px row (measured 2026-10-27, L7.54)
        if (keepSize && w && !w._mecGateSizers && w.type !== "hidden") {
            w._mecGateSizers = Object.fromEntries(_SIZERS.map((k) => [k, w[k]]));
        }
        _restoreWidget(w, w?._mecGateHeight ?? height);
        if (keepSize && w?._mecGateSizers) {
            for (const k of _SIZERS) {
                if (w._mecGateSizers[k] === undefined) delete w[k]; else w[k] = w._mecGateSizers[k];
            }
            delete w._mecGateSizers;
        }
    }
}

export function installModeGated(node, opts) {
    if (!node || !opts || typeof opts.installer !== "function") return;
    const activeWhen = typeof opts.activeWhen === "function" || Array.isArray(opts.activeWhen)
        ? opts.activeWhen : String(opts.activeWhen);
    const onActivate = typeof opts.onActivate === "function" ? opts.onActivate : null;
    const key = String(opts.installerKey || "default");
    const tag = `_mec_modegate_${key}`;
    const installer = opts.installer;
    const hostFinder = typeof opts.hostFinder === "function"
        ? opts.hostFinder
        : (() => null);
    const widgetFinder = typeof opts.widgetFinder === "function"
        ? opts.widgetFinder
        : (() => null);
    const widgetHeight = opts.widgetHeight ?? 460;
    const keepSize = opts.keepSize === true;

    const evaluate = () => {
        const mode = _modeValue(node);
        const isActive = _isActive(activeWhen, node, mode);
        const wh = typeof widgetHeight === "function" ? widgetHeight(node)
            : (typeof node._mecSplineEditWidgetH === "function" ? node._mecSplineEditWidgetH() : widgetHeight);
        if (isActive) {
            if (!node[tag]) {
                try {
                    installer(node);
                    node[tag] = true;
                } catch (err) {
                    console.error(`[MEC._mode_gate:${key}] installer failed:`, err);
                }
            }
            _showHost(hostFinder(node), widgetFinder(node), wh, keepSize);
            if (onActivate && node[tag]) {
                try { onActivate(node, mode); } catch (err) { __c2cReport("_mode_gate", err); }
            }
        } else {
            _hideHost(hostFinder(node), widgetFinder(node), keepSize);
        }
        if (typeof node.computeSize === "function" && typeof node.setSize === "function") {
            try {
                const sz = node.computeSize();
                if (Array.isArray(sz) && sz.length >= 2) node.setSize(sz);
            } catch (e) { __c2cReport("_mode_gate", e); }
        }
        if (typeof node.setDirtyCanvas === "function") {
            node.setDirtyCanvas(true, true);
        }
    };

    // Hook each watched widget only once per node (mode, plus any `watch` names).
    node._mecGateHooked = node._mecGateHooked || new Set();
    for (const name of ["mode", ...(Array.isArray(opts.watch) ? opts.watch : [])]) {
        if (node._mecGateHooked.has(name)) continue;
        const w = node.widgets?.find?.((x) => x && x.name === name);
        if (!w) continue;
        node._mecGateHooked.add(name);
        const orig = w.callback;
        w.callback = (v, ...rest) => {
            const r = orig?.call(w, v, ...rest);
            // Re-run ALL registered gate evaluators (active-mode one last).
            _runEvaluators(node);
            return r;
        };
    }
    node._mecGateEvaluators = node._mecGateEvaluators || [];
    node._mecGateEvaluators.push({ activeWhen, run: evaluate });

    // First evaluation after widgets settle. All installModeGated() calls for a
    // node run synchronously within nodeCreated, so by this tick every evaluator
    // is registered — run them ordered so the active editor wins the shared widget.
    setTimeout(() => _runEvaluators(node), 0);
}

export function getModeValue(node) { return _modeValue(node); }
