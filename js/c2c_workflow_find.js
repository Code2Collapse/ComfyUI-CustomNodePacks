// c2c_workflow_find.js — Ctrl+F in-workflow node finder (C2C)
// ---------------------------------------------------------------------
// Replaces ComfyUI's default Ctrl+F (which opens the *add-node* search
// palette) with a finder that searches nodes ALREADY PRESENT in the
// current graph. Fuzzy-matches across:
//   • node.title (user-overridable label)
//   • node.type / class_type
//   • node.properties.comment
//   • group labels containing the node
//   • node.id  (exact)
//
// Behavior:
//   Ctrl+F                → open finder, focus input
//   type query            → live ranked match list (top 12)
//   ↑/↓                   → move selection
//   Enter                 → pan + zoom to selected node, pulse-highlight
//   Tab                   → cycle next match in canvas (without closing)
//   Esc                   → close
//   When 0 matches OR query begins with "+", a "➕ Add node…" button
//   delegates to the stock Vue add-node palette.
//
// Settings: C2C ▸ Overlays ▸ "In-workflow finder (Ctrl+F)" (default ON).
// License: Apache-2.0
// ---------------------------------------------------------------------

import { app } from "../../scripts/app.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";
import { C } from './_c2c_theme.js';
import Fuse from "./_fuse.mjs";
import { capabilityFor } from "./c2c_node_taxonomy.js";

const STYLE_ID = "c2c-workflow-find-style";
const ROOT_ID  = "c2c-workflow-find-root";
const SETTING_ID = "c2c.workflowFind.enabled";
const HIGHLIGHT_MS = 1200;
const SYNONYMS_URL = new URL("./c2c_search_synonyms.json", import.meta.url).href;

// ── Synonym table (loaded async at setup) ─────────────────────────────
let _SYN = null;  // { term: [expansion, ...] }
let _synLoaded = false;
async function loadSynonyms() {
    if (_synLoaded) return _SYN;
    try {
        const r = await fetch(SYNONYMS_URL, { cache: "force-cache" });
        if (r.ok) {
            const json = await r.json();
            _SYN = Object.fromEntries(
                Object.entries(json).filter(([k]) => !k.startsWith("_"))
            );
        }
    } catch (e) {
        console.warn("[C2C.WorkflowFind] synonym table failed to load:", e);
    }
    _synLoaded = true;
    return _SYN;
}

/** Expand a raw query into a Fuse extended-search OR string. */
function expandQuery(q) {
    const raw = (q || "").trim();
    if (!raw) return "";
    if (!_SYN) return raw;
    // Split user query into whitespace-separated tokens; for each token,
    // collect its synonym expansions (lower-case key lookup).
    const tokens = raw.split(/\s+/).filter(Boolean);
    const variants = new Set([raw]);
    for (const t of tokens) {
        const key = t.toLowerCase();
        const exp = _SYN[key];
        if (Array.isArray(exp)) {
            for (const phrase of exp) {
                // Substitute this token within the original query so the
                // surrounding context is preserved.
                const rebuilt = tokens
                    .map((x) => (x === t ? phrase : x))
                    .join(" ");
                variants.add(rebuilt);
            }
        }
    }
    // Fuse extended search: '|' is logical OR between space-delimited
    // sub-queries. Quote multi-word phrases so they stay grouped.
    return [...variants]
        .map((v) => (v.includes(" ") ? `"${v}"` : v))
        .join(" | ");
}

function injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
#${ROOT_ID} {
    position: fixed; top: 56px; left: 50%; transform: translateX(-50%);
    z-index: var(--c2c-z-palette, 100001); width: min(520px, 92vw);
    background: rgba(24, 26, 30, 0.97);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px; box-shadow: 0 12px 40px rgba(0,0,0,0.55);
    font: 12.5px/1.4 ui-sans-serif, system-ui, "Segoe UI", sans-serif;
    color: var(--c2c-fgAltLight); backdrop-filter: blur(12px);
    padding: 8px 8px 6px;
}
#${ROOT_ID} .c2c-find-header {
    display:flex; align-items:center; gap:8px; padding: 0 4px 6px;
}
#${ROOT_ID} .c2c-find-title {
    font-weight: 600; color:var(--c2c-accentLink); letter-spacing:.2px;
    flex:0 0 auto;
}
#${ROOT_ID} .c2c-find-hint {
    color:var(--c2c-accentMuted); font-size: 11px; margin-left:auto;
}
#${ROOT_ID} input.c2c-find-input {
    width: 100%; box-sizing: border-box;
    background: rgba(0,0,0,0.35); color:var(--c2c-white);
    border: 1px solid rgba(255,255,255,0.10); border-radius: 7px;
    padding: 9px 10px; font: 13px ui-monospace, "Cascadia Code", monospace;
    outline: none;
}
#${ROOT_ID} input.c2c-find-input:focus {
    border-color: var(--c2c-accentSoft2); box-shadow: 0 0 0 2px rgba(91,141,239,0.18);
}
#${ROOT_ID} .c2c-find-results {
    margin-top: 6px; max-height: 320px; overflow-y: auto;
}
#${ROOT_ID} .c2c-find-row {
    display:flex; align-items:center; gap:8px;
    padding: 6px 8px; border-radius: 6px; cursor: pointer;
    user-select: none;
}
#${ROOT_ID} .c2c-find-row.sel { background: rgba(91,141,239,0.20); }
#${ROOT_ID} .c2c-find-row:hover { background: rgba(255,255,255,0.06); }
#${ROOT_ID} .c2c-find-row .ttl { color:var(--c2c-white); font-weight:500; }
#${ROOT_ID} .c2c-find-row .typ {
    color:var(--c2c-accentLink); font: 11px ui-monospace, monospace; opacity:.85;
}
#${ROOT_ID} .c2c-find-row .meta {
    color:var(--c2c-accentMuted); font-size: 10.5px; margin-left:auto;
}
#${ROOT_ID} .c2c-find-row mark {
    background: rgba(255, 209, 102, 0.32); color:var(--c2c-white);
    border-radius: 2px; padding: 0 1px;
}
#${ROOT_ID} .c2c-find-empty {
    color:var(--c2c-accentMuted); text-align:center; padding: 14px 0 10px;
}
#${ROOT_ID} .c2c-find-addbtn {
    display:block; width: 100%; margin-top:4px;
    padding: 8px; border-radius: 7px;
    background: rgba(91,141,239,0.16); color:var(--c2c-accentLight);
    border:1px solid rgba(91,141,239,0.35); cursor:pointer;
    font: 12px ui-sans-serif; text-align:center;
}
#${ROOT_ID} .c2c-find-addbtn:hover { background: rgba(91,141,239,0.28); }
.c2c-find-pulse {
    /* drawn by canvas overlay, not DOM — placeholder */
}`;
    document.head.appendChild(s);
}

// ── Fuzzy scoring (Fuse.js, with synonym-expanded extended search) ────
// Fuse v7 returns `matches:[{key,value,indices:[[a,b],...]}, ...]` when
// `includeMatches:true`. We convert that into the same {field, hits, val}
// shape the renderer already understands.

function highlight(text, hits) {
    if (!hits || !hits.length) return escapeHtml(text);
    // Merge overlapping/adjacent hit ranges so <mark> wrapping is clean.
    const ranges = [...hits].sort((a, b) => a[0] - b[0]);
    const merged = [];
    for (const [a, b] of ranges) {
        if (merged.length && a <= merged[merged.length - 1][1]) {
            merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], b);
        } else {
            merged.push([a, b]);
        }
    }
    let out = "", cur = 0;
    for (const [a, b] of merged) {
        out += escapeHtml(text.slice(cur, a));
        out += "<mark>" + escapeHtml(text.slice(a, b)) + "</mark>";
        cur = b;
    }
    out += escapeHtml(text.slice(cur));
    return out;
}

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;",
        '"': "&quot;", "'": "&#39;",
    }[c]));
}

// ── Graph search ──────────────────────────────────────────────────────
function getGroupForNode(node) {
    const groups = app.graph?._groups || app.graph?.groups || [];
    for (const g of groups) {
        if (typeof g.containsNode === "function" ? g.containsNode(node)
            : (g._nodes && g._nodes.includes(node))) {
            return g.title || "";
        }
    }
    return "";
}

function searchGraph(query) {
    const nodes = app.graph?._nodes || [];
    const q = (query || "").trim();
    if (!q) return [];

    // Build the row corpus on every search — graph is small (typically
    // < a few hundred nodes) and node titles/properties may be edited
    // between opens, so a cached index would go stale.
    const rows = nodes.map((n) => ({
        node: n,
        idStr: String(n.id ?? ""),
        title: n.title || n.type || "",
        type:  n.type || "",
        comment: n.properties?.comment || n.properties?.note || "",
        group: getGroupForNode(n),
        // Capability keywords describing what this node DOES — lets a
        // natural-language query ("upscale", "face swap", "caption")
        // match nodes whose name doesn't contain the search words.
        caps: capabilityFor(n.type),
    }));

    // Exact node-id hit gets a synthetic top-ranked result.
    const idExactRow = rows.find((r) => r.idStr === q);

    const fuse = new Fuse(rows, {
        keys: [
            { name: "title",   weight: 0.40 },
            { name: "type",    weight: 0.26 },
            { name: "caps",    weight: 0.14 },
            { name: "comment", weight: 0.12 },
            { name: "group",   weight: 0.08 },
        ],
        includeScore: true,
        includeMatches: true,
        threshold: 0.4,
        ignoreLocation: true,
        useExtendedSearch: true,
        minMatchCharLength: 1,
    });

    const pattern = expandQuery(q);
    const fuseHits = fuse.search(pattern, { limit: 24 });

    // Map Fuse output to the {score, hits, title, type, group} shape the
    // renderer expects. Fuse score: 0 = perfect, 1 = worst — invert for
    // our descending sort.
    const out = fuseHits.map((h) => {
        const r = h.item;
        // Pick the highest-weighted matched key to show with <mark>.
        const fieldOrder = ["title", "type", "comment", "group"];
        let best = null;
        for (const f of fieldOrder) {
            const m = h.matches?.find((x) => x.key === f);
            if (m) { best = m; break; }
        }
        return {
            node: r.node,
            score: Math.round((1 - (h.score ?? 1)) * 1000),
            hits: best
                // Fuse v7 indices are [start, end] INCLUSIVE; the
                // renderer + highlight() use half-open ranges, so we
                // convert once here.
                ? { field: best.key,
                    hits: (best.indices || []).map(([a, b]) => [a, b + 1]),
                    val: best.value }
                : null,
            title: r.title,
            type:  r.type,
            group: r.group,
            idExact: false,
        };
    });

    if (idExactRow && !out.some((r) => r.node === idExactRow.node)) {
        out.unshift({
            node: idExactRow.node, score: 9999,
            hits: { field: "title", hits: [], val: idExactRow.title },
            title: idExactRow.title, type: idExactRow.type, group: idExactRow.group,
            idExact: true,
        });
    }
    return out.slice(0, 24);
}

// ── Pan + zoom + pulse ────────────────────────────────────────────────
function focusNode(node) {
    if (!node) return;
    const canvas = app.canvas;
    if (!canvas) return;
    // Center node in viewport, zoom 1.0
    const ds = canvas.ds;
    if (ds) {
        const pad = 64;
        const cx = node.pos[0] + (node.size[0] / 2);
        const cy = node.pos[1] + (node.size[1] / 2);
        ds.scale = Math.min(1.2, Math.max(0.55, ds.scale || 1));
        ds.offset[0] = -cx + (canvas.canvas.width  / (2 * (window.devicePixelRatio || 1))) / ds.scale;
        ds.offset[1] = -cy + (canvas.canvas.height / (2 * (window.devicePixelRatio || 1))) / ds.scale;
    }
    canvas.selectNode?.(node, false);
    // pulse highlight via canvas
    node.flags = node.flags || {};
    node._c2c_pulse_until = performance.now() + HIGHLIGHT_MS;
    canvas.setDirty(true, true);
    if (!canvas._c2c_pulse_patched) {
        const origDrawNode = canvas.drawNode?.bind(canvas);
        if (origDrawNode) {
            canvas.drawNode = function (node, ctx) {
                const r = origDrawNode(node, ctx);
                if (node._c2c_pulse_until && performance.now() < node._c2c_pulse_until) {
                    const t = (HIGHLIGHT_MS - (node._c2c_pulse_until - performance.now())) / HIGHLIGHT_MS;
                    const a = 0.7 * (1 - t);
                    ctx.save();
                    ctx.strokeStyle = `rgba(91,141,239,${a.toFixed(3)})`;
                    ctx.lineWidth = 4 + 6 * t;
                    ctx.shadowColor = C.accentSoft2;
                    ctx.shadowBlur = 16;
                    ctx.strokeRect(-2, -2, node.size[0] + 4, node.size[1] + 4);
                    ctx.restore();
                    canvas.setDirty(true, true);
                } else if (node._c2c_pulse_until) {
                    delete node._c2c_pulse_until;
                }
                return r;
            };
            canvas._c2c_pulse_patched = true;
        }
    }
}

// ── Finder UI ─────────────────────────────────────────────────────────
let _root = null, _input = null, _list = null, _empty = null, _addbtn = null;
let _results = [], _sel = 0;

function ensureRoot() {
    if (_root) return _root;
    injectStyle();
    _root = document.createElement("div");
    _root.id = ROOT_ID;
    _root.style.display = "none";
    _root.innerHTML = `
        <div class="c2c-find-header">
            <div class="c2c-find-title">Find in workflow</div>
            <div class="c2c-find-hint">↑↓ select · Enter focus · Tab next · Esc close</div>
        </div>
        <input class="c2c-find-input" type="text" placeholder="Type to search nodes in this graph…" />
        <div class="c2c-find-results"></div>
        <div class="c2c-find-empty" style="display:none">No matching nodes in this workflow.</div>
        <button class="c2c-find-addbtn" style="display:none">➕ Add a new node…</button>
    `;
    document.body.appendChild(_root);
    _input  = _root.querySelector("input.c2c-find-input");
    _list   = _root.querySelector(".c2c-find-results");
    _empty  = _root.querySelector(".c2c-find-empty");
    _addbtn = _root.querySelector(".c2c-find-addbtn");

    _input.addEventListener("input", refresh);
    _input.addEventListener("keydown", onKey);
    _addbtn.addEventListener("click", () => {
        close();
        // Delegate to stock add-node palette. Prefer the Vue command;
        // fall back to the legacy LiteGraph search (made visible by
        // c2c_legacy_search_visibility.js). NEVER dispatch a Ctrl+F
        // key event here — that would loop straight back to us.
        try {
            const mgr = app.extensionManager?.command;
            if (mgr && typeof mgr.execute === "function") {
                // Workspace.SearchBox.Toggle is the canonical add-node
                // palette command in modern ComfyUI. (Comfy.NewSearch
                // never shipped under that id.)
                mgr.execute("Workspace.SearchBox.Toggle");
                return;
            }
        } catch { /* fall through */ }
        try {
            // Legacy path: call the canvas search box directly.
            app.canvas?.showSearchBox?.(new MouseEvent("mousedown"));
        } catch (e) {
            console.warn("[C2C.WorkflowFind] could not open add-node search:", e);
        }
    });
    return _root;
}

function open(prefill) {
    ensureRoot();
    _root.style.display = "block";
    _input.value = typeof prefill === "string" ? prefill : "";
    refresh();
    _input.focus();
    if (_input.value) {
        // Place caret at end so the user can edit/extend the prefilled query.
        _input.setSelectionRange(_input.value.length, _input.value.length);
    }
}

function close() {
    if (_root) _root.style.display = "none";
}

function refresh() {
    const q = _input.value;
    _results = searchGraph(q);
    _sel = 0;
    render();
}

function render() {
    if (!_results.length) {
        _list.innerHTML = "";
        _empty.style.display = _input.value ? "block" : "none";
        _addbtn.style.display = _input.value ? "block" : "none";
        return;
    }
    _empty.style.display = "none";
    _addbtn.style.display = "none";
    const rows = _results.map((r, i) => {
        const cls = i === _sel ? "c2c-find-row sel" : "c2c-find-row";
        const fld = r.hits?.field || "title";
        const ttl = (fld === "title")
            ? highlight(r.title || "(untitled)", r.hits?.hits)
            : escapeHtml(r.title || "(untitled)");
        const typ = (fld === "type")
            ? highlight(r.type, r.hits?.hits)
            : escapeHtml(r.type);
        const meta = r.group
            ? `<span class="meta">▣ ${escapeHtml(r.group)} · #${r.node.id}</span>`
            : `<span class="meta">#${r.node.id}</span>`;
        return `<div class="${cls}" data-i="${i}">
            <span class="ttl">${ttl}</span>
            <span class="typ">${typ}</span>
            ${meta}
        </div>`;
    }).join("");
    _list.innerHTML = rows;
    [..._list.children].forEach((el) => {
        el.addEventListener("click", () => {
            _sel = +el.dataset.i;
            const r = _results[_sel];
            if (r) { focusNode(r.node); close(); }
        });
    });
}

function onKey(ev) {
    if (ev.key === "Escape") { ev.preventDefault(); close(); return; }
    if (ev.key === "ArrowDown") {
        ev.preventDefault();
        _sel = Math.min(_results.length - 1, _sel + 1);
        render();
        return;
    }
    if (ev.key === "ArrowUp") {
        ev.preventDefault();
        _sel = Math.max(0, _sel - 1);
        render();
        return;
    }
    if (ev.key === "Enter") {
        ev.preventDefault();
        const r = _results[_sel];
        if (r) { focusNode(r.node); close(); }
        return;
    }
    if (ev.key === "Tab") {
        ev.preventDefault();
        if (!_results.length) return;
        _sel = (_sel + (ev.shiftKey ? -1 : 1) + _results.length) % _results.length;
        render();
        const r = _results[_sel];
        if (r) focusNode(r.node);  // peek but keep finder open
        return;
    }
}

// ── Global key intercepts ─────────────────────────────────────────────
// Tiny helper: is the finder root currently visible?
function isOpen() {
    return !!(_root && _root.style.display !== "none"
              && getComputedStyle(_root).display !== "none");
}

function onGlobalKey(ev) {
    // ─ Esc anywhere closes the finder, even if focus has drifted off the
    //   input (clicked outside, etc.). Bound on capture so we win before
    //   ComfyUI's own Esc handlers can swallow it.
    if (ev.key === "Escape" && isOpen()) {
        ev.preventDefault();
        ev.stopPropagation();
        close();
        return;
    }
    // ─ Ctrl+F / Cmd+F: open, focus, or TOGGLE close when already open.
    if (!(ev.ctrlKey || ev.metaKey)) return;
    if (ev.key !== "f" && ev.key !== "F") return;
    const enabled = (() => {
        try { return app.ui.settings.getSettingValue(SETTING_ID, true) !== false; }
        catch { return true; }
    })();
    if (!enabled) return;
    const ae = document.activeElement;
    const tag = ae?.tagName?.toLowerCase();
    const isEditable = tag === "input" || tag === "textarea"
                       || ae?.isContentEditable === true;
    // Pressing Ctrl+F again while focus is inside our finder → close it.
    if (isOpen() && _root && _root.contains(ae)) {
        ev.preventDefault();
        ev.stopPropagation();
        close();
        return;
    }
    // Focus drifted away but finder is still visible → bring focus back
    // (don't close — user might want to keep typing where they left off).
    if (isOpen()) {
        ev.preventDefault();
        ev.stopPropagation();
        _input?.focus();
        return;
    }
    if (isEditable) return;  // user is typing in a widget — leave Ctrl+F alone
    ev.preventDefault();
    ev.stopPropagation();
    open();
}

// Click-outside-closes: if the user mouses anywhere outside the finder
// while it's open, dismiss it. Matches Spotlight / VS Code Cmd-P UX.
function onGlobalPointerDown(ev) {
    if (!isOpen()) return;
    if (_root && _root.contains(ev.target)) return;
    close();
}

app.registerExtension({
    name: "C2C.WorkflowFind",
    async setup() {
        try {
            app.ui.settings.addSetting({
                id: SETTING_ID,
                name: "In-workflow finder (Ctrl+F searches THIS graph)",
                tooltip:
                    "When enabled, Ctrl+F opens a fuzzy finder over the " +
                    "nodes already on the canvas. Disable to restore the " +
                    "stock ComfyUI add-node search.",
                type: "boolean",
                defaultValue: true,
                category: ["c2c", "Overlays", "Workflow Find"],
            });
        } catch (e) { __c2cReport("c2c_workflow_find", e); }
        // Kick off synonym table load — search still works without it.
        loadSynonyms().catch(() => { /* already logged */ });
        window.addEventListener("keydown", onGlobalKey, true);
        // Bound on capture so we see clicks before LiteGraph swallows them.
        window.addEventListener("mousedown", onGlobalPointerDown, true);
        console.log("[C2C.WorkflowFind] Ctrl+F now searches nodes in workflow (Fuse).");
    },

    /**
     * Right-click on a node → "🔍 Find similar nodes…" entry that
     * pre-fills the in-workflow finder with the clicked node's type.
     * Survives across LiteGraph versions; only adds the entry when the
     * setting is enabled.
     */
    getNodeMenuOptions(_canvas, options, node) {
        try {
            const enabled = (() => {
                try { return app.ui.settings.getSettingValue(SETTING_ID, true) !== false; }
                catch { return true; }
            })();
            if (!enabled || !node) return options;
            const seed = node.type || node.title || "";
            if (!seed) return options;
            // Insert near the top, before the default "Properties" group.
            const entry = {
                content: "🔍 Find similar nodes…",
                callback: () => { try { open(seed); } catch (e) { __c2cReport("c2c_workflow_find", e); } },
            };
            // LiteGraph passes options as an array; the first nullish slot
            // is a separator placeholder — insert just above it.
            const sepIdx = options.findIndex((o) => o == null);
            if (sepIdx > 0) options.splice(sepIdx, 0, entry);
            else options.unshift(entry, null);
        } catch (e) { __c2cReport("c2c_workflow_find", e); }
        return options;
    },
});
