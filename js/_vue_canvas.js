// _vue_canvas.js — renderer-aware canvas widget install.
//
// LiteGraph "custom" widgets (node.addCustomWidget with a draw(ctx,…) +
// mouse() pair) are NOT rendered under ComfyUI's Nodes 2.0 / Vue renderer —
// Vue never calls draw(), so these editors show up blank there. This adapter
// keeps the CLASSIC path byte-identical (it just calls addCustomWidget) and,
// only when Vue nodes is active, additionally mounts a real <canvas> DOM
// widget that drives the SAME draw()/mouse() logic. The classic renderer —
// the default, and what everything is tuned for — is therefore never touched.
//
// Usage in a widget file, replacing `node.addCustomWidget(widget);`:
//     import { installCanvasWidget } from "../_vue_canvas.js";
//     installCanvasWidget(node, widget, 320);   // 320 = drawn height in px

function _vueActive() {
    try {
        return window.app?.ui?.settings?.getSettingValue?.("Comfy.VueNodes.Enabled") === true;
    } catch (_) {
        return false;
    }
}

export function installCanvasWidget(node, widget, height) {
    // Classic renderer: identical to the original call — zero behaviour change.
    node.addCustomWidget(widget);
    if (!_vueActive()) return widget;

    // Vue renderer: hide the custom widget from Vue (canvasOnly skips it) and
    // mount a DOM <canvas> that runs the widget's own draw()/mouse().
    widget.options = widget.options || {};
    widget.options.canvasOnly = true;          // Vue's shouldRenderAsVue() skips it
    widget.computeSize = () => [0, -4];         // don't double-count height

    const host = document.createElement("div");
    host.style.cssText = `width:100%;height:${height}px;position:relative;overflow:hidden;`;
    const cvs = document.createElement("canvas");
    cvs.style.cssText = "width:100%;height:100%;display:block;outline:none;touch-action:none;";
    cvs.tabIndex = 0;
    host.appendChild(cvs);

    let domWidget;
    try {
        domWidget = node.addDOMWidget(widget.name + "_vue", "canvas", host, {
            serialize: false,
            getMinHeight: () => height,
            getHeight: () => height,
        });
    } catch (_) {
        return widget;   // addDOMWidget unavailable — classic path already ran
    }

    const ctx = cvs.getContext("2d");
    let raf = 0;

    const localXY = (e) => {
        const r = cvs.getBoundingClientRect();
        const kx = (r.width || 1);
        // widget.size[0] is kept in sync with the CSS width below, so the
        // mouse handler's own scaling (canvas.w / widget.size[0]) stays valid.
        const x = (e.clientX - r.left) * ((widget.size?.[0] || kx) / kx);
        const y = (e.clientY - r.top) * ((height) / (r.height || height));
        return [x, y];
    };

    const forward = (e, type) => {
        try {
            widget.mouse?.({ type, button: e.button, buttons: e.buttons,
                             shiftKey: e.shiftKey, ctrlKey: e.ctrlKey,
                             altKey: e.altKey, metaKey: e.metaKey,
                             deltaY: e.deltaY, deltaX: e.deltaX },
                           localXY(e), node);
        } catch (_) { /* handler error must not kill the loop */ }
    };
    cvs.addEventListener("pointerdown", (e) => { cvs.focus(); forward(e, "pointerdown"); e.stopPropagation(); });
    cvs.addEventListener("pointermove", (e) => forward(e, "pointermove"));
    cvs.addEventListener("pointerup",   (e) => forward(e, "pointerup"));
    cvs.addEventListener("wheel",       (e) => forward(e, "wheel"), { passive: true });
    cvs.addEventListener("contextmenu", (e) => e.preventDefault());

    const tick = () => {
        // Node gone → stop.
        if (node.graph == null) { cancelAnimationFrame(raf); return; }
        const r = cvs.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const w = Math.max(1, Math.round(r.width || 320));
        const pw = Math.round(w * dpr), ph = Math.round(height * dpr);
        if (cvs.width !== pw || cvs.height !== ph) { cvs.width = pw; cvs.height = ph; }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, height);
        widget.size = [w, height];          // keep the mouse handler's scaling correct
        try { widget.draw(ctx, node, w, 0, height); } catch (_) { /* keep looping */ }
        raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    const origRemoved = node.onRemoved;
    node.onRemoved = function (...a) {
        cancelAnimationFrame(raf);
        try { host.remove(); } catch (_) {}
        return origRemoved?.apply(this, a);
    };

    return widget;
}
