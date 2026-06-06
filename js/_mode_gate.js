import { reportFailure as __c2cReport } from "./_c2c_report.js";

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
// =====================================================================

const GATE_HOOKED = new WeakSet();

function _modeValue(node) {
    const w = node?.widgets?.find?.((w) => w && w.name === "mode");
    return w ? String(w.value ?? "") : "";
}

function _collapseWidget(w) {
    if (!w) return;
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
    w.type = undefined;
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

function _hideHost(host, widget) {
    if (host && host.style) host.style.display = "none";
    _collapseWidget(widget);
}

function _showHost(host, widget, height) {
    if (host && host.style) host.style.display = "";
    _restoreWidget(widget, height);
}

export function installModeGated(node, opts) {
    if (!node || !opts || typeof opts.installer !== "function") return;
    const activeWhen = String(opts.activeWhen);
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

    const evaluate = () => {
        const mode = _modeValue(node);
        const isActive = (mode === activeWhen);
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
            _showHost(hostFinder(node), widgetFinder(node), wh);
        } else {
            _hideHost(hostFinder(node), widgetFinder(node));
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

    // Hook mode-widget changes only once per node.
    if (!GATE_HOOKED.has(node)) {
        GATE_HOOKED.add(node);
        const modeW = node.widgets?.find?.((w) => w && w.name === "mode");
        if (modeW) {
            const orig = modeW.callback;
            modeW.callback = (v, ...rest) => {
                const r = orig?.call(modeW, v, ...rest);
                // Re-run ALL registered gate evaluators for this node.
                const evals = node._mecGateEvaluators || [];
                for (const fn of evals) {
                    try { fn(); } catch (e) { __c2cReport("_mode_gate", e); }
                }
                return r;
            };
        }
    }
    node._mecGateEvaluators = node._mecGateEvaluators || [];
    node._mecGateEvaluators.push(evaluate);

    // First evaluation after widgets settle.
    setTimeout(evaluate, 0);
}

export function getModeValue(node) { return _modeValue(node); }
