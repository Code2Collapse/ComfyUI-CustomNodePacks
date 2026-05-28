/**
 * mec_complexity_hud.js — Phase 5: Workflow Complexity Meter HUD
 *
 * Floating chip in the corner that classifies the open graph as
 * Easy / Medium / Advanced based on node count and link density.
 * Updates live whenever the graph changes.
 *
 * Setting:
 *   mec.complexity_hud.enabled — bool (default true)
 */

import { app } from "../../scripts/app.js";

const HUD_ID   = "mec-complexity-hud";
const STYLE_ID = "mec-complexity-hud-style";
const POP_ID   = "mec-complexity-pop";

const TIER = {
    EASY:     { label: "🌱 Easy",     color: "var(--c2c-okSoft)", bg: "var(--c2c-okBgDark)" },
    MEDIUM:   { label: "⚙ Medium",    color: "var(--c2c-yellow)", bg: "var(--c2c-warnBg)" },
    ADVANCED: { label: "🔥 Advanced", color: "var(--c2c-red)", bg: "var(--c2c-dangerBg)" },
};

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
#${HUD_ID} {
    position: fixed;
    /* Iter-4c: anchored to bottom-center so it never overlaps the
       ComfyUI status/queue bar in the bottom-left nor the workflow
       tabs / RGthree progress bar at the top. */
    bottom: 8px;
    left: 50%;
    transform: translateX(-50%);
    z-index: var(--c2c-z-hud, 1000);
    padding: 4px 12px;
    border-radius: 14px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 11px;
    font-weight: 600;
    border: 1px solid transparent;
    color: var(--c2c-fg);
    background: var(--c2c-bg);
    cursor: pointer;
    user-select: none;
    box-shadow: 0 2px 6px rgba(0,0,0,0.4);
    transition: background 0.2s, color 0.2s, border-color 0.2s, transform 0.08s;
    pointer-events: auto;
    opacity: 0.92;
}
#${HUD_ID}:hover { opacity: 1; }
#${HUD_ID}:active { transform: translateX(-50%) scale(0.96); }
#${HUD_ID} .mec-cx-detail {
    font-weight: 400;
    color: var(--c2c-slate400);
    margin-left: 6px;
}
#${HUD_ID}:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.6);
}
#${POP_ID} {
    position: fixed;
    display: none;
    z-index: var(--c2c-z-popover, 9000);
    background: var(--c2c-panelBg, var(--c2c-bg));
    color: var(--c2c-fg);
    border: 1px solid var(--c2c-surface0);
    border-radius: 10px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.55);
    padding: 10px 12px;
    font: 11px/1.35 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    min-width: 260px; max-width: 360px;
}
#${POP_ID}.open { display: block; }
#${POP_ID} h4 {
    margin: 0 0 6px 0; font-size: 10px; color: var(--c2c-slate400);
    text-transform: uppercase; letter-spacing: 0.4px;
}
#${POP_ID} .cx-row {
    display: flex; justify-content: space-between; gap: 12px;
    padding: 3px 0; border-bottom: 1px dashed rgba(255,255,255,0.05);
}
#${POP_ID} .cx-row:last-of-type { border-bottom: 0; }
#${POP_ID} .cx-row.err  { color: var(--c2c-red); }
#${POP_ID} .cx-row.warn { color: var(--c2c-peach); }
#${POP_ID} .cx-row .v   { font-weight: 700; font-variant-numeric: tabular-nums; }
#${POP_ID} .cx-tier-badge {
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-weight: 700; margin-bottom: 6px;
}
#${POP_ID} .cx-items {
    margin-top: 6px; max-height: 180px; overflow-y: auto;
    border-top: 1px solid var(--c2c-surface0); padding-top: 6px;
}
#${POP_ID} .cx-item {
    padding: 2px 4px; cursor: pointer; border-radius: 4px;
    display: flex; gap: 6px; align-items: center;
}
#${POP_ID} .cx-item:hover { background: rgba(137,180,250,0.12); }
#${POP_ID} .cx-tag {
    font-size: 9px; padding: 1px 5px; border-radius: 4px;
    text-transform: uppercase; letter-spacing: 0.04em;
    background: var(--c2c-surface0); color: var(--c2c-fg);
}
#${POP_ID} .cx-tag.cycle    { background: var(--c2c-red); color: var(--c2c-bg); }
#${POP_ID} .cx-tag.dead     { background: var(--c2c-peach); color: var(--c2c-bg); }
#${POP_ID} .cx-tag.dangling { background: var(--c2c-mauve); color: var(--c2c-bg); }
    `.trim();
    document.head.appendChild(style);
}

let _popEl = null;
let _popOpen = false;

function _ensurePop() {
    if (_popEl && document.body.contains(_popEl)) return _popEl;
    _popEl = document.createElement("div");
    _popEl.id = POP_ID;
    _popEl.setAttribute("role", "dialog");
    _popEl.setAttribute("aria-label", "Complexity & graph health");
    document.body.appendChild(_popEl);
    return _popEl;
}

function _ghCounts() {
    try {
        const gh = window.__C2C_GRAPH_HEALTH__;
        if (!gh) return { available: false };
        return {
            available: true,
            dead: (gh.dead || []).length,
            cycles: (gh.cycles || []).length,
            dangling: (gh.dangling || []).length,
            items: Array.isArray(gh.items) ? gh.items.slice(0, 30) : [],
        };
    } catch { return { available: false }; }
}

function _renderPop(tier, nodes, links) {
    const pop = _ensurePop();
    const gh = _ghCounts();
    const esc = (s) => String(s ?? "").replace(/[&<>]/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[m]));
    pop.innerHTML = `
        <div class="cx-tier-badge" style="background:${tier.bg};color:${tier.color};border:1px solid ${tier.color};">
            ${tier.label}
        </div>
        <h4>Workflow size</h4>
        <div class="cx-row"><span>Nodes</span><span class="v">${nodes}</span></div>
        <div class="cx-row"><span>Links</span><span class="v">${links}</span></div>
        <h4 style="margin-top:8px">Graph health</h4>
        ${gh.available ? `
            <div class="cx-row ${gh.cycles ? "err" : ""}"><span>Cycles</span><span class="v">${gh.cycles}</span></div>
            <div class="cx-row ${gh.dead ? "warn" : ""}"><span>Dead nodes</span><span class="v">${gh.dead}</span></div>
            <div class="cx-row ${gh.dangling ? "warn" : ""}"><span>Dangling inputs</span><span class="v">${gh.dangling}</span></div>
            ${gh.items.length ? `<div class="cx-items">${gh.items.map((r) => `
                <div class="cx-item" data-id="${esc(String(r.id))}">
                    <span class="cx-tag ${esc(r.tag)}">${esc(r.tag)}</span>
                    <span>${esc(r.text || "")}</span>
                </div>`).join("")}</div>` : ""}
        ` : `<div style="opacity:0.55;">Graph health analyser not loaded.</div>`}
    `;
    pop.querySelectorAll(".cx-item").forEach((el) => {
        el.addEventListener("click", () => {
            const id = el.dataset.id;
            try { window.__C2C_GRAPH_HEALTH__?.focus?.(id); } catch { /* */ }
        });
    });
}

function _positionPop() {
    if (!_popEl) return;
    const hud = document.getElementById(HUD_ID);
    if (!hud) return;
    const r = hud.getBoundingClientRect();
    const w = _popEl.offsetWidth || 280;
    const h = _popEl.offsetHeight || 200;
    // Prefer popping ABOVE the HUD when the HUD itself sits near the
    // viewport bottom (iter-4b moved it to bottom-left). Fall back to
    // below if there isn't enough room above.
    const spaceBelow = window.innerHeight - r.bottom;
    const spaceAbove = r.top;
    if (spaceBelow >= h + 12 || spaceBelow >= spaceAbove) {
        _popEl.style.top = (r.bottom + 6) + "px";
    } else {
        _popEl.style.top = Math.max(8, r.top - h - 6) + "px";
    }
    // center the popover under (or over) the hud bar
    let left = r.left + r.width / 2 - w / 2;
    if (left < 8) left = 8;
    if (left + w > window.innerWidth - 8) left = window.innerWidth - w - 8;
    _popEl.style.left = left + "px";
}

let _outsideHandler = null;
let _escHandler = null;
function _openPop(tier, nodes, links) {
    _ensurePop();
    _renderPop(tier, nodes, links);
    _popEl.classList.add("open");
    _popOpen = true;
    requestAnimationFrame(_positionPop);
    if (!_outsideHandler) {
        _outsideHandler = (ev) => {
            if (!_popEl) return;
            if (_popEl.contains(ev.target)) return;
            const hud = document.getElementById(HUD_ID);
            if (hud && hud.contains(ev.target)) return;
            _closePop();
        };
        document.addEventListener("mousedown", _outsideHandler, true);
    }
    if (!_escHandler) {
        _escHandler = (ev) => { if (ev.key === "Escape" && _popOpen) _closePop(); };
        document.addEventListener("keydown", _escHandler, true);
    }
    window.addEventListener("resize", _positionPop, { passive: true });
}

function _closePop() {
    if (_popEl) _popEl.classList.remove("open");
    _popOpen = false;
    if (_outsideHandler) {
        document.removeEventListener("mousedown", _outsideHandler, true);
        _outsideHandler = null;
    }
    if (_escHandler) {
        document.removeEventListener("keydown", _escHandler, true);
        _escHandler = null;
    }
    window.removeEventListener("resize", _positionPop);
}

function _ensureHud() {
    let hud = document.getElementById(HUD_ID);
    if (!hud) {
        hud = document.createElement("div");
        hud.id = HUD_ID;
        hud.title = "Workflow complexity & graph health — click for details";
        hud.addEventListener("click", (ev) => {
            ev.stopPropagation();
            if (_popOpen) { _closePop(); return; }
            const { nodes, links } = _countGraph();
            const tier = _classify(nodes, links);
            _openPop(tier, nodes, links);
        });
        document.body.appendChild(hud);
    }
    return hud;
}

function _classify(nodeCount, linkCount) {
    if (nodeCount <= 8 && linkCount <= 12) return TIER.EASY;
    if (nodeCount <= 25 && linkCount <= 40) return TIER.MEDIUM;
    return TIER.ADVANCED;
}

function _countGraph() {
    const g = app.graph;
    if (!g) return { nodes: 0, links: 0 };
    const nodes = g._nodes ? g._nodes.length : (g.nodes?.length || 0);
    // links is a flat object/array of LiteGraph LLink entries
    let links = 0;
    if (g.links) {
        if (Array.isArray(g.links)) {
            for (const l of g.links) if (l) links++;
        } else {
            links = Object.keys(g.links).length;
        }
    }
    return { nodes, links };
}

let _scheduled = false;
function _scheduleUpdate() {
    if (_scheduled) return;
    _scheduled = true;
    requestAnimationFrame(() => {
        _scheduled = false;
        _update();
    });
}

function _update() {
    const enabled = (() => {
        try { return app.ui.settings.getSettingValue("mec.complexity_hud.enabled", true); }
        catch { return true; }
    })();
    const hud = _ensureHud();
    if (!enabled) {
        hud.style.display = "none";
        return;
    }
    hud.style.display = "block";

    const { nodes, links } = _countGraph();
    const tier = _classify(nodes, links);
    hud.style.background   = tier.bg;
    hud.style.color        = tier.color;
    hud.style.borderColor  = tier.color;
    hud.innerHTML = `${tier.label}<span class="mec-cx-detail">${nodes} nodes · ${links} links</span>`;
    // Iteration 2: publish complexity into window.C2CStatusStrip so the
    // OmniPill Stats section's all-stats slot renders it. The legacy
    // #mec-complexity-hud chip is CSS-hidden by _c2c_theme.js.
    try {
        const strip = (typeof window !== "undefined") ? window.C2CStatusStrip : null;
        if (strip && typeof strip.register === "function") {
            const stateMap = { EASY: "ok", MEDIUM: "warn", ADVANCED: "err" };
            const tierKey = (tier === TIER.EASY) ? "EASY" : (tier === TIER.MEDIUM ? "MEDIUM" : "ADVANCED");
            strip.register({
                id: "cmplx",
                label: "CX",
                value: `${tier.label.replace(/[^A-Za-z0-9 ]/g, "").trim()} · ${nodes}n`,
                state: stateMap[tierKey] || "idle",
                tooltip: `Workflow complexity — ${tier.label} (${nodes} nodes / ${links} links)`,
                priority: 50,
            });
        }
    } catch (e) {
        console.warn("[C2CComplexityHUD] publish to strip failed:", e);
    }
}

function _hookGraphMutations() {
    const g = app.graph;
    if (!g) return;

    // LiteGraph fires these on every relevant edit
    const hookOne = (obj, prop, after) => {
        if (typeof obj[prop] !== "function") return;
        const orig = obj[prop];
        if (orig._mecCxWrapped) return;
        obj[prop] = function (...args) {
            const r = orig.apply(this, args);
            try { after(); } catch { /* swallow */ }
            return r;
        };
        obj[prop]._mecCxWrapped = true;
    };

    hookOne(g, "add",          _scheduleUpdate);
    hookOne(g, "remove",       _scheduleUpdate);
    hookOne(g, "configure",    _scheduleUpdate);
    hookOne(g, "clear",        _scheduleUpdate);
    if (typeof g.onNodeAdded   !== "function") g.onNodeAdded   = _scheduleUpdate;
    if (typeof g.onNodeRemoved !== "function") g.onNodeRemoved = _scheduleUpdate;
    if (typeof g.onConnectionChange !== "function") g.onConnectionChange = _scheduleUpdate;
}

app.registerExtension({
    name: "C2C.ComplexityHUD",
    settings: [
        {
            id: "mec.complexity_hud.enabled",
            name: "Complexity HUD: enabled",
            tooltip: "Show the Easy/Medium/Advanced complexity chip at the top of the canvas.",
            type: "boolean",
            default: true,
            onChange: () => _update(),
        },
    ],
    async setup() {
        _injectStyle();
        _ensureHud();
        _hookGraphMutations();
        _update();

        // Periodic safety refresh — in case some custom op mutates the graph
        // without going through the hooked methods. Store the handle so the
        // interval can be torn down (e.g. on hot-reload of the extension or
        // when the page navigates away).
        const _t = setInterval(_update, 2000);
        window.addEventListener("beforeunload", () => clearInterval(_t), { once: true });
        window.__MEC_COMPLEXITY_HUD_INTERVAL = _t;

        console.log("[MEC.ComplexityHUD] Loaded.");
    },
});
