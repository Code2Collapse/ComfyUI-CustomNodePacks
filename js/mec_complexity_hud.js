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

const TIER = {
    EASY:     { label: "🌱 Easy",     color: "#a6e3a1", bg: "#1e3a2a" },
    MEDIUM:   { label: "⚙ Medium",    color: "#f9e2af", bg: "#3a361e" },
    ADVANCED: { label: "🔥 Advanced", color: "#f38ba8", bg: "#3a1e29" },
};

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
#${HUD_ID} {
    position: fixed;
    top: 8px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 99996;
    padding: 4px 12px;
    border-radius: 14px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 11px;
    font-weight: 600;
    border: 1px solid transparent;
    color: #cdd6f4;
    background: #1e1e2e;
    cursor: default;
    user-select: none;
    box-shadow: 0 2px 6px rgba(0,0,0,0.4);
    transition: background 0.2s, color 0.2s, border-color 0.2s;
    pointer-events: auto;
}
#${HUD_ID} .mec-cx-detail {
    font-weight: 400;
    color: #94a3b8;
    margin-left: 6px;
}
#${HUD_ID}:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.6);
}
    `.trim();
    document.head.appendChild(style);
}

function _ensureHud() {
    let hud = document.getElementById(HUD_ID);
    if (!hud) {
        hud = document.createElement("div");
        hud.id = HUD_ID;
        hud.title = "Workflow complexity (node count + link density)";
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
    name: "MEC.ComplexityHUD",
    settings: [
        {
            id: "mec.complexity_hud.enabled",
            name: "Complexity HUD: enabled",
            tooltip: "Show the Easy/Medium/Advanced complexity chip at the top of the canvas.",
            type: "boolean",
            defaultValue: true,
            onChange: () => _update(),
        },
    ],
    async setup() {
        _injectStyle();
        _ensureHud();
        _hookGraphMutations();
        _update();

        // Periodic safety refresh — in case some custom op mutates the graph
        // without going through the hooked methods.
        setInterval(_update, 2000);

        console.log("[MEC.ComplexityHUD] Loaded.");
    },
});
