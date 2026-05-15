/**
 * mec_token_counter.js — Phase 10: Prompt Token Counter
 *
 * On CLIPTextEncode / CLIPTextEncodeSDXL / SDXLPromptStyler etc., shows a
 * live token-count badge under the multiline STRING widget. Updates with a
 * 250 ms debounce. Calls /mec/token_count for the exact CLIP count; uses a
 * client-side heuristic while the server is responding.
 *
 * Setting:
 *   mec.token_counter.enabled — bool (default true)
 */

import { app } from "../../scripts/app.js";

const TEXT_NODE_RE = /cliptextencode|smz_cliptextencode|promptstyler|advanced.*clip.*text/i;
const TEXT_WIDGET_NAMES = ["text", "prompt", "positive", "negative", "string"];
const LIMIT = 77;
const DEBOUNCE_MS = 250;
const STYLE_ID = "mec-token-counter-style";

const _heuristic = (txt) => {
    if (!txt) return 0;
    const parts = String(txt).split(/[\s.,;:!?()[\]{}"'`<>/\\]+/).filter(Boolean);
    return Math.round(parts.length * 1.3);
};

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.mec-token-badge {
    font-family: monospace;
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 3px;
    background: #313244;
    color: #cdd6f4;
    display: inline-block;
    margin-top: 2px;
    user-select: none;
}
.mec-token-badge.ok   { color: #a6e3a1; }
.mec-token-badge.warn { color: #f9e2af; }
.mec-token-badge.over { color: #f38ba8; background: #3a1e29; }
.mec-token-badge.estim { opacity: 0.7; }
    `.trim();
    document.head.appendChild(style);
}

async function _exactCount(text) {
    try {
        const resp = await fetch("/mec/token_count", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });
        const json = await resp.json();
        if (json.success && json.data) return json.data;
    } catch { /* ignore */ }
    return null;
}

function _classify(count) {
    if (count > LIMIT)       return "over";
    if (count > LIMIT * 0.8) return "warn";
    return "ok";
}

function _formatLabel(count, exact) {
    return `${count} / ${LIMIT} tokens${exact ? "" : " (≈)"}`;
}

function _attachToNode(node) {
    if (!node || node._mecTokenCounterAttached) return;
    if (!TEXT_NODE_RE.test(node.type || node.comfyClass || "")) return;
    const widget = (node.widgets || []).find(w =>
        w && (w.type === "customtext" || w.type === "string" || w.type === "STRING") &&
        TEXT_WIDGET_NAMES.includes((w.name || "").toLowerCase())
    );
    if (!widget) return;
    node._mecTokenCounterAttached = true;

    // Add a non-clickable widget after the text widget that displays the badge.
    const badgeWidget = {
        type: "mec_token_badge",
        name: "tokens",
        value: "0 / 77",
        _state: { count: 0, klass: "ok", exact: false, t: null },
        draw(ctx, _node, widget_width, y, _h) {
            const text = this.value;
            const klass = this._state.klass;
            ctx.save();
            ctx.font = "10px monospace";
            const pad = 4;
            const w = ctx.measureText(text).width + pad * 2;
            ctx.fillStyle = klass === "over" ? "#3a1e29" : "#313244";
            ctx.fillRect(8, y, w, 14);
            ctx.fillStyle = klass === "over" ? "#f38ba8"
                          : klass === "warn" ? "#f9e2af"
                          : "#a6e3a1";
            ctx.textBaseline = "middle";
            ctx.fillText(text, 8 + pad, y + 7);
            ctx.restore();
        },
        computeSize() { return [120, 16]; },
    };
    if (!node.widgets) node.widgets = [];
    node.widgets.push(badgeWidget);

    const update = () => {
        const enabled = (() => {
            try { return app.ui.settings.getSettingValue("mec.token_counter.enabled", true); }
            catch { return true; }
        })();
        if (!enabled) {
            badgeWidget.value = "";
            return;
        }
        const txt = widget.value || "";
        // Heuristic first (instant)
        const est = _heuristic(txt);
        badgeWidget._state.count = est;
        badgeWidget._state.klass = _classify(est);
        badgeWidget._state.exact = false;
        badgeWidget.value = _formatLabel(est, false);
        node.setDirtyCanvas?.(true, true);

        clearTimeout(badgeWidget._state.t);
        badgeWidget._state.t = setTimeout(async () => {
            const exact = await _exactCount(txt);
            if (exact) {
                badgeWidget._state.count = exact.tokens;
                badgeWidget._state.klass = _classify(exact.tokens);
                badgeWidget._state.exact = !!exact.exact;
                badgeWidget.value = _formatLabel(exact.tokens, exact.exact);
                node.setDirtyCanvas?.(true, true);
            }
        }, DEBOUNCE_MS);
    };

    // Hook input changes
    const origCallback = widget.callback;
    widget.callback = function (v, ...rest) {
        const r = origCallback ? origCallback.call(this, v, ...rest) : undefined;
        update();
        return r;
    };

    // Also react when value is set programmatically via configure()
    update();
}

function _scanAll() {
    const g = app.graph;
    if (!g || !g._nodes) return;
    for (const n of g._nodes) _attachToNode(n);
}

app.registerExtension({
    name: "MEC.TokenCounter",
    settings: [
        {
            id: "mec.token_counter.enabled",
            name: "Token Counter: CLIP token badge on prompt widgets",
            tooltip: "Show a live x/77 token-count under CLIPTextEncode prompts.",
            type: "boolean",
            defaultValue: true,
            onChange: () => app.canvas?.setDirty?.(true, true),
        },
    ],
    async nodeCreated(node) {
        _attachToNode(node);
    },
    async setup() {
        _injectStyle();
        _scanAll();
        // Store the handle so the interval can be torn down on page unload.
        const _t = setInterval(_scanAll, 4000);  // catch nodes added by graph load
        window.addEventListener("beforeunload", () => clearInterval(_t), { once: true });
        window.__MEC_TOKEN_COUNTER_INTERVAL = _t;
        console.log("[MEC.TokenCounter] Loaded.");
    },
});
