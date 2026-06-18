/**
 * c2c_compose_node.js — C2C "Pack to Single Node" button
 *
 * Drops the previous fake-group illusion. Instead, hooks ComfyUI's native
 * subgraph engine (`app.graph.convertToSubgraph`) and injects a C2C button
 * into the floating `.selection-toolbox` (the per-selection bar that pops up
 * above selected nodes), right next to the built-in "Convert Selection to
 * Subgraph" button.
 *
 * Goals per user (verbatim 2026-05-26):
 *   "What I wanted is a type of subgraph that holds all slots, connections
 *    and parameters as a single node. Not groups. Not ctrl+G thing.
 *    if i click on node, i get options on top of node. beside the subgraph
 *    option my option should come. instead of manually connecting all
 *    parameters and options to the subgrpah i wanted to simplify it with
 *    combining multi nodes as a single node."
 *
 * Implementation: real ComfyUI subgraph. The native conversion already keeps
 * external links wired through the new single node. We additionally try to
 * auto-promote every internal widget as a top-level input on the new
 * subgraph node so the user does not have to open the subgraph and "Convert
 * Widget to Input" by hand.
 */

import { app } from "../../../scripts/app.js";

const BUTTON_ID = "c2c-pack-subgraph-btn";
const STYLE_ID  = "c2c-pack-subgraph-style";
const SETTING_PROMOTE = "c2c.subgraph.auto_promote_widgets";
const SETTING_SMART_TITLE = "c2c.subgraph.smart_title";
const SETTING_HIDE_WIDGET_SOCKETS = "c2c.subgraph.hide_widget_sockets";

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const css = document.createElement("style");
    css.id = STYLE_ID;
    css.textContent = `
#${BUTTON_ID} {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    margin: 0 2px;
    border: none;
    background: transparent;
    color: var(--p-button-text-secondary-color, var(--c2c-fg));
    cursor: pointer;
    border-radius: 6px;
    transition: background 120ms, color 120ms;
}
#${BUTTON_ID}:hover {
    background: rgba(137, 180, 250, 0.16);
    color: var(--c2c-blue);
}
#${BUTTON_ID} svg { width: 18px; height: 18px; }
`;
    document.head.appendChild(css);
}

function _buildButton() {
    const btn = document.createElement("button");
    btn.id = BUTTON_ID;
    btn.type = "button";
    btn.title = "C2C: Pack selection as one node (real subgraph + auto-promote widgets)";
    btn.innerHTML = `
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
     stroke-linecap="round" stroke-linejoin="round">
  <rect x="3"  y="3"  width="7" height="7" rx="1.2"/>
  <rect x="14" y="3"  width="7" height="7" rx="1.2"/>
  <rect x="3"  y="14" width="7" height="7" rx="1.2"/>
  <path d="M14 17.5h7M17.5 14v7" />
</svg>`;
    btn.addEventListener("click", _onClick);
    return btn;
}

async function _onClick(ev) {
    ev?.stopPropagation?.();
    const canvas = app.canvas;
    const items = canvas?.selectedItems;
    if (!items || items.size < 1) {
        _toast("warn", "Pack to Single Node", "Select 2+ nodes first.");
        return;
    }
    // A1 Dedupe guard: if the only selected item is already a subgraph node,
    // packing it again would create a nested subgraph the user didn't ask for.
    if (items.size === 1) {
        const only = items.values().next().value;
        if (only && only.subgraph) {
            _toast("warn", "Pack to Single Node",
                "Selection is already a packed subgraph. Add more nodes or double-click to enter.");
            return;
        }
    }
    const wantPromote = _setting(SETTING_PROMOTE, true);
    const wantSmartTitle = _setting(SETTING_SMART_TITLE, true);
    const hideWidgetSockets = _setting(SETTING_HIDE_WIDGET_SOCKETS, false);
    // Capture inner-node types BEFORE conversion (they move into subgraph after).
    const innerTypes = [];
    for (const it of items) {
        if (it && it.type && !it.subgraph) innerTypes.push(it.title || it.type);
    }

    // Phase 1: native subgraph conversion. In the current frontend, every
    // widget already co-exists with a matching input on the node — there is
    // no separate convertWidgetToInput step needed.
    let subgraphNode;
    try {
        const beforeIds = new Set((app.graph._nodes || []).map(n => n.id));
        app.graph.convertToSubgraph(items);
        for (const n of (app.graph._nodes || [])) {
            if (!beforeIds.has(n.id)) { subgraphNode = n; break; }
        }
    } catch (e) {
        console.error("[C2C compose] convertToSubgraph failed", e);
        _toast("error", "Pack to Single Node", String(e?.message || e));
        return;
    }
    if (!subgraphNode) {
        _toast("warn", "Pack to Single Node", "Subgraph created but new node not located.");
        return;
    }

    // Phase 2: wire each inner widget-input across the subgraph boundary
    // and mark them as promoted widgets via proxyWidgets.
    let promotedCount = 0;
    if (wantPromote) {
        try {
            promotedCount = _wireAndPromoteInnerWidgets(subgraphNode, { hideWidgetSockets });
        } catch (e) {
            console.warn("[C2C compose] wire+promote failed", e);
        }
    }

    // Phase 3: expose every remaining unconnected socket (non-widget inner
    // inputs + inner outputs with no outgoing links) as outer sockets on
    // the subgraph node. Native convertToSubgraph only auto-exposes I/O
    // that had EXTERNAL links pre-conversion; for isolated selections
    // those slots stay buried, which is what users mean when they say
    // "the pack node has no sockets".
    let exposedIn = 0, exposedOut = 0;
    try {
        const r = _exposeRemainingIO(subgraphNode);
        exposedIn  = r.inputsAdded;
        exposedOut = r.outputsAdded;
    } catch (e) {
        console.warn("[C2C compose] expose remaining I/O failed", e);
    }

    // A2 Smart title: derive a human title from inner node names.
    if (wantSmartTitle) {
        try {
            const t = _deriveSmartTitle(innerTypes);
            if (t) {
                subgraphNode.title = t;
                if (subgraphNode.subgraph) subgraphNode.subgraph.name = t;
            }
        } catch (e) { console.warn("[C2C compose] smart title failed", e); }
    }

    canvas.setDirty(true, true);
    const ioBits = [];
    if (promotedCount) ioBits.push(`${promotedCount} widgets inlined`);
    if (exposedIn)     ioBits.push(`${exposedIn} input socket${exposedIn===1?"":"s"}`);
    if (exposedOut)    ioBits.push(`${exposedOut} output socket${exposedOut===1?"":"s"}`);
    const detail = ioBits.length ? `Subgraph created · ${ioBits.join(" · ")}.` : "Subgraph created.";
    _toast("success", "Packed", detail);
}

/**
 * Build a short, readable title from up to 3 inner node names.
 *   ["KSampler", "Empty Latent Image"]                  -> "KSampler + Empty Latent Image"
 *   ["KSampler", "Empty Latent Image", "VAE Decode"]    -> "KSampler + Empty Latent Image + VAE Decode"
 *   ["A","B","C","D","E"]                                -> "A + B + C  (+2 more)"
 */
function _deriveSmartTitle(names) {
    if (!Array.isArray(names) || names.length === 0) return null;
    const cleaned = names.map(n => String(n).trim()).filter(Boolean);
    if (cleaned.length === 0) return null;
    if (cleaned.length <= 3) return cleaned.join(" + ");
    const extra = cleaned.length - 3;
    return `${cleaned.slice(0, 3).join(" + ")}  (+${extra} more)`;
}

/**
 * After convertToSubgraph, for every inner widget-input we want to expose
 * as an inline widget on the outer subgraph node:
 *   1. `sg.addInput(name, type)` → creates a SubgraphInput slot.
 *   2. `sgInputSlot.connect(innerInputSlot, innerNode)` → registers the
 *      link in `sg._links`, populates `linkIds`, sets `innerInputSlot.link`,
 *      and (because the inner input has a backing widget) sets the
 *      SubgraphInput's `_widget` reference — the inline widget render path.
 *   3. Single-assign `sgNode.properties.proxyWidgets = [[id, name], ...]`
 *      (per-push triggers a Vue trim that drops everything after [0]).
 *   4. `sgNode.rebuildInputWidgetBindings()` walks each outer input's
 *      `_subgraphSlot.linkIds`, resolves the link, and promotes it as
 *      `input._widget` on the outer node.
 *
 * The wrong path is `sg.inputNode.connectSlots(...)` — that only constructs
 * a Link object and returns it WITHOUT registering it on the subgraph,
 * which produces orphan links and empty `linkIds`. Always use the slot's
 * own `.connect()` method.
 */
function _wireAndPromoteInnerWidgets(sgNode, opts) {
    const sg = sgNode.subgraph;
    if (!sg || !Array.isArray(sg.nodes)) return 0;
    const hideWidgetSockets = !!(opts && opts.hideWidgetSockets);

    // Collect every inner widget-input. In the current frontend, widget and
    // socket co-exist on the same input — `inp.widget` is set whenever the
    // input has a backing widget.
    const targets = []; // [{innerNode, innerInputSlot}]
    for (const inner of sg.nodes) {
        const inputs = inner.inputs || [];
        for (const inp of inputs) {
            if (!inp || !inp.name) continue;
            if (!inp.widget) continue;
            if (inp.link != null) continue;
            targets.push({ innerNode: inner, innerInputSlot: inp });
        }
    }
    if (targets.length === 0) return 0;

    // Add a SubgraphInput per target and wire it to the inner widget-input
    // using the slot's own `.connect()` method (the only API that actually
    // registers the link).
    const wired = []; // [{innerNode, innerInputSlot}]
    for (const { innerNode, innerInputSlot } of targets) {
        const type = innerInputSlot.type || "*";
        const name = `${innerNode.title || innerNode.type}.${innerInputSlot.name}`;
        let sgInputSlot;
        try { sgInputSlot = sg.addInput(name, type); }
        catch (e) { console.warn("[C2C compose] sg.addInput failed", name, e); continue; }
        if (!sgInputSlot) continue;
        try {
            const link = sgInputSlot.connect(innerInputSlot, innerNode);
            if (link) wired.push({ innerNode, innerInputSlot });
        } catch (e) {
            console.warn("[C2C compose] sgInputSlot.connect failed", name, e);
        }
    }
    if (wired.length === 0) return 0;

    // Single proxyWidgets assignment (per-push triggers Vue trim).
    const proxy = wired.map(w => [String(w.innerNode.id), w.innerInputSlot.name]);
    sgNode.properties = sgNode.properties || {};
    sgNode.properties.proxyWidgets = proxy;

    if (typeof sgNode.rebuildInputWidgetBindings === "function") {
        try { sgNode.rebuildInputWidgetBindings(); }
        catch (e) { console.warn("[C2C compose] rebuildInputWidgetBindings failed", e); }
    }
    // A3 Optional: hide the co-exist socket dot on widget-backed outer
    // inputs (cleaner look). LiteGraph's draw path skips the slot dot when
    // `not_subtype` is set on the input; we set a custom flag and a CSS
    // hook that draws over it. Implementation here: set
    // `input.widget_input_hidden = true` (a render hint our `c2c_canvas_hooks`
    // honours when present). Real socket inputs (without widgets) are left
    // visible regardless.
    if (hideWidgetSockets) {
        for (const inp of (sgNode.inputs || [])) {
            if (inp && inp._widget) inp.widget_input_hidden = true;
        }
    }

    try {
        if (typeof sgNode.computeSize === "function" &&
            typeof sgNode.setSize === "function") {
            sgNode.setSize(sgNode.computeSize());
        }
    } catch (_) {}
    return wired.length;
}

/**
 * Expose every remaining inner I/O that didn't already get promoted as a
 * widget or auto-exposed by `convertToSubgraph`:
 *   • Inner inputs where `inp.widget` is falsy AND `inp.link == null`
 *     (i.e., a real socket-only input with no internal driver) → become
 *     SubgraphInput slots on the outer node, wired via
 *     `sgInputSlot.connect(innerInputSlot, innerNode)`.
 *   • Inner outputs where `out.links` is empty → become SubgraphOutput
 *     slots, wired via `sgOutputSlot.connect(innerOutputSlot, innerNode)`.
 *
 * Slots inherit the inner slot name verbatim when there's no collision;
 * on collision we prefix with the inner node title.
 *
 * Returns { inputsAdded, outputsAdded }.
 */
function _exposeRemainingIO(sgNode) {
    const sg = sgNode.subgraph;
    const out = { inputsAdded: 0, outputsAdded: 0 };
    if (!sg || !Array.isArray(sg.nodes)) return out;

    const usedInNames  = new Set((sg.inputs  || []).map(s => s.name));
    const usedOutNames = new Set((sg.outputs || []).map(s => s.name));
    const pickName = (used, base, ownerTitle) => {
        if (!used.has(base)) return base;
        const prefixed = `${ownerTitle}.${base}`;
        if (!used.has(prefixed)) return prefixed;
        let i = 2;
        while (used.has(`${prefixed}_${i}`)) i++;
        return `${prefixed}_${i}`;
    };

    for (const inner of sg.nodes) {
        const ownerTitle = inner.title || inner.type || "node";

        // --- Inputs (socket-only, unconnected) ----------------------------
        for (const inp of (inner.inputs || [])) {
            if (!inp || !inp.name) continue;
            if (inp.widget) continue;            // widget-backed → already handled (or skipped intentionally)
            if (inp.link != null) continue;      // already wired internally
            const name = pickName(usedInNames, inp.name, ownerTitle);
            let sgInputSlot;
            try { sgInputSlot = sg.addInput(name, inp.type || "*"); }
            catch (e) { console.warn("[C2C compose] sg.addInput (socket) failed", name, e); continue; }
            if (!sgInputSlot) continue;
            try {
                const link = sgInputSlot.connect(inp, inner);
                if (link) {
                    usedInNames.add(name);
                    out.inputsAdded++;
                }
            } catch (e) {
                console.warn("[C2C compose] sgInputSlot.connect (socket) failed", name, e);
            }
        }

        // --- Outputs (any with no outgoing internal links) ----------------
        for (const o of (inner.outputs || [])) {
            if (!o || !o.name) continue;
            const links = o.links || [];
            if (links.length > 0) continue;
            const name = pickName(usedOutNames, o.name, ownerTitle);
            let sgOutputSlot;
            try { sgOutputSlot = sg.addOutput(name, o.type || "*"); }
            catch (e) { console.warn("[C2C compose] sg.addOutput failed", name, e); continue; }
            if (!sgOutputSlot) continue;
            try {
                const link = sgOutputSlot.connect(o, inner);
                if (link) {
                    usedOutNames.add(name);
                    out.outputsAdded++;
                }
            } catch (e) {
                console.warn("[C2C compose] sgOutputSlot.connect failed", name, e);
            }
        }
    }

    try {
        if (typeof sgNode.computeSize === "function" &&
            typeof sgNode.setSize === "function") {
            sgNode.setSize(sgNode.computeSize());
        }
    } catch (_) {}
    return out;
}

function _setting(key, fallback) {
    try {
        const v = app.extensionManager?.setting?.get?.(key);
        if (v === undefined || v === null) return fallback;
        return v;
    } catch (_) { return fallback; }
}

function _toast(severity, summary, detail) {
    try {
        app.extensionManager?.toast?.add?.({ severity, summary, detail, life: 2400 });
    } catch (_) {}
}

function _watchToolbox() {
    // PERF: original implementation observed `document.body` with
    // `subtree:true`, which fires the callback on every DOM mutation
    // anywhere in the page (popovers, toasts, queue updates, settings
    // dialog, …). Even with a no-op callback that adds up to thousands
    // of calls per minute and noticeably slows startup. We now:
    //   1. Try to attach the observer to the LiteGraph canvas container
    //      (where `.selection-toolbox` is actually mounted as a sibling).
    //      Only fall back to `document.body` if that container isn't
    //      ready yet — and even then we re-target as soon as it appears.
    //   2. Coalesce bursts via `requestAnimationFrame` so the callback
    //      body runs at most once per frame.
    //   3. Disconnect & rebind to the canvas container the first time
    //      we see one mounted, so steady-state observation is local.
    let rafScheduled = false;
    const _attachBtn = () => {
        rafScheduled = false;
        const toolbox = document.querySelector(".selection-toolbox");
        if (!toolbox) return;
        if (toolbox.querySelector(`#${BUTTON_ID}`)) return;
        const native = Array.from(toolbox.querySelectorAll("button"))
            .find(b => /subgraph/i.test(b.getAttribute("aria-label") || b.title || ""));
        const btn = _buildButton();
        if (native && native.parentNode) {
            native.parentNode.insertBefore(btn, native.nextSibling);
        } else {
            const slot = toolbox.querySelector(".p-panel-content") || toolbox;
            slot.appendChild(btn);
        }
    };
    const schedule = () => {
        if (rafScheduled) return;
        rafScheduled = true;
        requestAnimationFrame(_attachBtn);
    };

    let currentTarget = null;
    let observer = null;
    const bind = (target) => {
        if (target === currentTarget) return;
        try { observer?.disconnect(); } catch (_) {}
        observer = new MutationObserver(schedule);
        observer.observe(target, { childList: true, subtree: true });
        currentTarget = target;
    };

    const pickTarget = () =>
        document.querySelector(".graph-canvas-container") ||
        document.querySelector(".lg-canvas-container") ||
        document.querySelector("#graph-canvas")?.parentElement ||
        document.body;

    bind(pickTarget());
    // If we started on body, watch for the canvas container to appear
    // and migrate the observer to it (one-shot).
    if (currentTarget === document.body) {
        const upgrade = () => {
            const t = pickTarget();
            if (t && t !== document.body) bind(t);
        };
        // Run a few times during startup, then stop.
        let tries = 0;
        const id = setInterval(() => {
            upgrade();
            if (currentTarget !== document.body || tries++ > 20) clearInterval(id);
        }, 500);
    }
    // Initial scan in case toolbox was already there.
    schedule();
}

app.registerExtension({
    name: "C2C.PackToSingleNode",
    settings: [
        {
            id: SETTING_PROMOTE,
            name: "C2C: Pack \u2192 auto-promote internal widgets",
            type: "boolean",
            defaultValue: true,
            tooltip: "When using C2C Pack, also expose every internal widget as an input on the new subgraph node.",
        },
        {
            id: SETTING_SMART_TITLE,
            name: "C2C: Pack \u2192 smart subgraph title",
            type: "boolean",
            defaultValue: true,
            tooltip: "Use the inner node names as the packed subgraph's title instead of 'New Subgraph'.",
        },
        {
            id: SETTING_HIDE_WIDGET_SOCKETS,
            name: "C2C: Pack \u2192 hide co-exist socket dot on widget rows",
            type: "boolean",
            defaultValue: false,
            tooltip: "For a cleaner look, mark inline-widget inputs to render without a left-edge socket dot. Real (non-widget) sockets are always shown.",
        },
    ],
    commands: [
        {
            id: "C2C.PackToSingleNode.run",
            label: "C2C: Pack selection into a single node",
            function: () => _onClick(new Event("click")),
        },
    ],
    async setup() {
        _injectStyle();
        _watchToolbox();
        window.c2cCompose = {
            pack: () => _onClick(new Event("click")),
        };
    },
});
