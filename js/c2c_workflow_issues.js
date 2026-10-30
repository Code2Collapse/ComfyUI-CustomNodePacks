/**
 * c2c_workflow_issues.js - the live workflow validator and its Workflow Issues button (L9.71, owner A20 17.1).
 *
 * The validator (moved unchanged from c2c_workflow_wizard.js, P7) scans the graph, subgraphs included, for COMBO
 * values that no longer exist and required inputs left unconnected. Issues surface as a "⚠ N workflow issues" button
 * that opens a fix-it list.
 *
 * The button used to sit at a fixed spot on top of LiteGraph's canvas info and the console. Now it is a floating
 * control the user places: drag it (a click below 5 px of movement still opens the list), Alt + arrow keys move it,
 * right-click resets or hides it. Its position is saved in ComfyUI's UI settings (never in the workflow) as a fraction
 * of the viewport, and it always steps out of the console's way and the canvas info's way without overwriting where
 * the user put it. Design: docs/design/workflow_issues_button.md. Geometry: js/_c2c_float_place.js.
 */
import { app } from "../../scripts/app.js";
import { forAllNodes } from "./_subgraph_walk.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";
import { getRuntime } from "./_c2c_runtime.js";
import { TOOLBAR_GUTTER } from "./_c2c_window.js";
import { avoidRects, clampRect, fromFraction, snapToEdges, toFraction } from "./_c2c_float_place.js";

const PILL_ID = "c2c-wizard-validator-pill";             // ids unchanged: scripts and user CSS find them
const LIST_ID = "c2c-wizard-validator-list";
const S_SHOW = "c2c.issues.button.show";
const S_SIZE = "c2c.issues.button.size";
const S_STATE = "c2c.issues.button.state";               // hidden: { pos: {fx, fy}, panel: {fx, fy, w, h} }
const CMD = {
    SHOW: "C2C.WorkflowIssues.ShowButton",
    HIDE: "C2C.WorkflowIssues.HideButton",
    RESET: "C2C.WorkflowIssues.ResetPosition",
    OPEN: "C2C.WorkflowIssues.OpenList",
};
const DRAG_PX = 5;
const KEY_STEP = 16;
const Z = 990;                                           // above the canvas, below PrimeVue menus (1000) / dialogs (1100)
const SIZES = {
    small: "600 10px ui-sans-serif,system-ui;padding:3px 9px",
    normal: "600 11px ui-sans-serif,system-ui;padding:4px 12px",
    large: "600 13px ui-sans-serif,system-ui;padding:6px 15px",
};

let _objInfo = null;
let _issues = [];
let _timer = 0;
let _drag = null;
let _suppressClick = false;
let _reported = false;

// ── settings ────────────────────────────────────────────────────────────────
function getS(id, fallback) {
    try {
        const v = app.extensionManager?.setting?.get?.(id) ?? app.ui?.settings?.getSettingValue?.(id);
        return v === undefined || v === null ? fallback : v;
    } catch (_e) { return fallback; }
}

function setS(id, value) {
    try {
        const s = app.extensionManager?.setting;
        const r = s?.set ? s.set(id, value) : app.ui?.settings?.setSettingValueAsync?.(id, value);
        r?.catch?.((err) => { if (!_reported) { _reported = true; __c2cReport("c2c_workflow_issues", err); } });
    } catch (err) {
        if (!_reported) { _reported = true; __c2cReport("c2c_workflow_issues", err); }
    }
}

function state() {
    const s = getS(S_STATE, null);
    return s && typeof s === "object" ? s : {};
}

function saveState(patch) { setS(S_STATE, { ...state(), ...patch }); }

// ── validator (P7, unchanged) ───────────────────────────────────────────────
async function _defs() {
    if (_objInfo) return _objInfo;
    try {
        const r = await fetch("/object_info");
        _objInfo = await r.json();
    } catch (_) { _objInfo = {}; }
    return _objInfo;
}

function _isLinkType(spec) {
    const t = Array.isArray(spec) ? spec[0] : spec;
    return typeof t === "string" && /^[A-Z][A-Z0-9_]*$/.test(t) &&
        !["INT", "FLOAT", "STRING", "BOOLEAN"].includes(t);
}

async function _scan() {
    const defs = await _defs();
    const out = [];
    const all = [];
    forAllNodes((x) => { all.push(x); });
    for (const n of all) {
        const def = defs[n.comfyClass || n.type];
        if (!def) continue;
        const req = (def.input && def.input.required) || {};
        // (a) stale combo values
        for (const w of (n.widgets || [])) {
            const spec = req[w.name];
            if (!spec || !Array.isArray(spec) || !Array.isArray(spec[0])) continue;
            const opts = spec[0];
            if (opts.length && w.value != null && !opts.includes(w.value)) {
                out.push({
                    kind: "stale_combo", node: n, widget: w, opts,
                    msg: `${n.title || n.type}: "${w.name}" is set to a file that no longer exists (${String(w.value).slice(0, 40)})`,
                    fix: `Set to ${String(opts[0]).slice(0, 32)}`,
                    apply: () => { w.value = opts[0]; try { w.callback?.(opts[0], app.canvas, n); } catch (_) {} n.setDirtyCanvas?.(true, true); },
                });
            }
        }
        // (b) unconnected required link inputs
        for (const [name, spec] of Object.entries(req)) {
            if (!_isLinkType(spec)) continue;
            const slot = (n.inputs || []).find((i) => i.name === name);
            if (slot && slot.link == null) {
                out.push({
                    kind: "missing_link", node: n,
                    msg: `${n.title || n.type}: required input "${name}" is not connected`,
                    fix: "Show node",
                    apply: () => { try { app.canvas.centerOnNode(n); app.canvas.selectNode(n); } catch (_) {} },
                });
            }
        }
    }
    _issues = out;
    _render();
}

function _schedule() {
    if (_timer) clearTimeout(_timer);
    _timer = setTimeout(() => { _timer = 0; _scan().catch(() => {}); }, 700);
}

// ── geometry ────────────────────────────────────────────────────────────────
function viewport() { return { w: window.innerWidth, h: window.innerHeight, top: TOOLBAR_GUTTER }; }

function shown(el) {
    if (!el?.isConnected) return false;
    const cs = getComputedStyle(el);
    if (cs.display === "none" || cs.visibility === "hidden") return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
}

/**
 * LiteGraph's canvas info block (T / I / N / V / FPS): six 13 px lines at `fpsInfoLocation` (frontend 1.53 sets
 * [side-toolbar width, null] = past the toolbar, at the bottom; the <canvas> itself runs under the toolbar from x 0).
 */
function canvasInfoRect() {
    const c = app.canvas;
    const cv = c?.canvas;
    if (!cv || !c.show_info) return null;
    const r = cv.getBoundingClientRect();
    if (!r.width || !r.height) return null;
    const lines = (c.graph ? 5 : 1) + (c.info_text ? 1 : 0);
    const h = (lines + 1) * 13 + 8;
    const loc = c.fpsInfoLocation || [];
    const x = r.left + (Number(loc[0]) || 10);
    const y = Number.isFinite(loc[1]) && loc[1] ? r.top + loc[1] : r.bottom - h;
    return { left: x - 4, top: y, right: x + 130, bottom: y + h };
}

/** Core's side toolbar and an open sidebar tab (node library, queue, ...): visible ones on the left. */
function sidebarRects() {
    const out = [];
    for (const el of document.querySelectorAll(".side-toolbar-container, .side-bar-panel")) {
        if (!shown(el)) continue;
        const r = el.getBoundingClientRect();
        out.push({ left: r.left, top: r.top, right: r.right, bottom: r.bottom });
    }
    return out;
}

/** What the button must never cover: the console (core's bottom panel), the canvas info and the sidebars. */
function keepOut() {
    const out = [];
    for (const el of document.querySelectorAll(".bottom-panel")) {
        if (!shown(el)) continue;
        const r = el.getBoundingClientRect();
        out.push({ left: r.left - 4, top: r.top - 4, right: r.right + 4, bottom: r.bottom + 4 });
    }
    const info = canvasInfoRect();
    if (info) out.push(info);
    return out.concat(sidebarRects());
}

/** The canvas's bottom-left, past the side toolbar, just above the canvas info. */
function defaultBox(w, h) {
    const vp = viewport();
    const cv = app.canvas?.canvas?.getBoundingClientRect?.();
    const inset = Math.max(cv && cv.width ? cv.left : 0,
        ...sidebarRects().filter((r) => r.left < vp.w / 2).map((r) => r.right), 0);
    const bottom = cv && cv.height ? cv.bottom : vp.h;
    const info = canvasInfoRect();
    return { x: inset + 12, y: (info ? info.top : bottom) - h - 8, w, h };
}

function pillSize(p) {
    const r = p.getBoundingClientRect();
    return { w: Math.ceil(r.width) || 150, h: Math.ceil(r.height) || 24 };
}

/** Where the button is drawn: the user's spot (or the default), clamped, then moved out of the keep-out areas. */
function placePill() {
    const p = document.getElementById(PILL_ID);
    if (!p || p.style.display === "none" || _drag) return;
    const { w, h } = pillSize(p);
    const vp = viewport();
    const want = fromFraction(state().pos, w, h, vp) || defaultBox(w, h);
    const box = avoidRects(want, keepOut(), vp);
    p.style.left = `${Math.round(box.x)}px`;
    p.style.top = `${Math.round(box.y)}px`;
}

// ── the button ──────────────────────────────────────────────────────────────
function _pill() {
    let p = document.getElementById(PILL_ID);
    if (p) return p;
    p = document.createElement("button");
    p.id = PILL_ID;
    p.type = "button";
    p.dataset.mecDock = "1";                             // c2c_dock_anchor.js: our own HUD, not an occluder
    styleP(p);
    p.addEventListener("pointerdown", onDown);
    p.addEventListener("pointermove", onMove);
    p.addEventListener("pointerup", onUp);
    p.addEventListener("pointercancel", onUp);
    p.addEventListener("click", (e) => {
        if (_suppressClick) { _suppressClick = false; e.preventDefault(); return; }
        _toggleList();
    });
    p.addEventListener("keydown", onKey);
    p.addEventListener("contextmenu", (e) => { e.preventDefault(); openMenu(e.clientX, e.clientY); });
    document.body.appendChild(p);
    return p;
}

function styleP(p) {
    const font = SIZES[getS(S_SIZE, "normal")] || SIZES.normal;
    const display = p.style.display || "none";
    p.style.cssText =
        `position:fixed;left:0;top:0;z-index:${Z};display:${display};touch-action:none;user-select:none;` +
        `border-radius:999px;border:1px solid var(--c2c-dangerBg);background:var(--c2c-bg2);color:var(--c2c-red);` +
        `font:${font};cursor:grab;box-shadow:0 2px 8px rgba(0,0,0,.4);`;
}

function onDown(e) {
    if (e.button !== 0) return;
    const r = e.currentTarget.getBoundingClientRect();
    _drag = { id: e.pointerId, sx: e.clientX, sy: e.clientY, ox: r.left, oy: r.top, w: r.width, h: r.height, moved: false };
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch (_e) { /* synthetic events */ }
}

function onMove(e) {
    if (!_drag || e.pointerId !== _drag.id) return;
    const dx = e.clientX - _drag.sx, dy = e.clientY - _drag.sy;
    if (!_drag.moved && Math.hypot(dx, dy) < DRAG_PX) return;
    _drag.moved = true;
    const p = e.currentTarget;
    p.style.cursor = "grabbing";
    const box = clampRect({ x: _drag.ox + dx, y: _drag.oy + dy, w: _drag.w, h: _drag.h }, viewport());
    p.style.left = `${Math.round(box.x)}px`;
    p.style.top = `${Math.round(box.y)}px`;
}

function onUp(e) {
    if (!_drag || e.pointerId !== _drag.id) return;
    const p = e.currentTarget;
    try { p.releasePointerCapture(e.pointerId); } catch (_e) { /* not captured */ }
    const d = _drag;
    _drag = null;
    p.style.cursor = "grab";
    if (!d.moved || e.type === "pointercancel") { if (e.type === "pointercancel") placePill(); return; }
    _suppressClick = true;                               // the click that follows a drag is not a click
    setTimeout(() => { _suppressClick = false; }, 0);
    const r = p.getBoundingClientRect();
    dropAt({ x: r.left, y: r.top, w: r.width, h: r.height });
}

/** A user placement: kept out of the console / canvas info, snapped to near edges, saved, drawn. */
function dropAt(box) {
    const vp = viewport();
    const final = avoidRects(snapToEdges(clampRect(box, vp), vp), keepOut(), vp);
    saveState({ pos: toFraction(final, vp) });
    const p = document.getElementById(PILL_ID);
    if (p) { p.style.left = `${Math.round(final.x)}px`; p.style.top = `${Math.round(final.y)}px`; }
}

function onKey(e) {
    const step = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
    if (step && e.altKey) {
        e.preventDefault();
        const r = e.currentTarget.getBoundingClientRect();
        const k = e.shiftKey ? KEY_STEP * 4 : KEY_STEP;
        const vp = viewport();
        const box = avoidRects(clampRect({ x: r.left + step[0] * k, y: r.top + step[1] * k, w: r.width, h: r.height }, vp), keepOut(), vp);
        saveState({ pos: toFraction(box, vp) });
        e.currentTarget.style.left = `${Math.round(box.x)}px`;
        e.currentTarget.style.top = `${Math.round(box.y)}px`;
    } else if (e.key === "Escape") {
        document.getElementById(LIST_ID)?.remove();
    }
}

function label(p) {
    const n = _issues.length;
    p.textContent = `⚠ ${n} workflow issue${n > 1 ? "s" : ""}`;
    p.title = `${n} workflow issue${n > 1 ? "s" : ""}: click for the list. Drag to move it (Alt + arrow keys too); `
        + "right-click to reset or hide it.";
    p.setAttribute("aria-label", `Workflow issues: ${n}. Open the list`);
}

// ── right-click menu ────────────────────────────────────────────────────────
function openMenu(x, y) {
    document.getElementById("c2c-issues-menu")?.remove();
    const m = document.createElement("div");
    m.id = "c2c-issues-menu";
    m.setAttribute("role", "menu");
    m.style.cssText =
        `position:fixed;left:${x}px;top:${y}px;z-index:${Z + 2};min-width:190px;padding:4px;border-radius:8px;` +
        "background:var(--c2c-bg);border:1px solid var(--c2c-surface2);box-shadow:0 8px 24px rgba(0,0,0,.5);" +
        "font:12px ui-sans-serif,system-ui;color:var(--c2c-fg);";
    const item = (text, fn) => {
        const b = document.createElement("button");
        b.type = "button";
        b.setAttribute("role", "menuitem");
        b.textContent = text;
        b.style.cssText = "display:block;width:100%;text-align:left;padding:6px 10px;border:0;border-radius:6px;"
            + "background:transparent;color:inherit;font:inherit;cursor:pointer;";
        b.addEventListener("mouseenter", () => { b.style.background = "var(--c2c-surface0)"; });
        b.addEventListener("mouseleave", () => { b.style.background = "transparent"; });
        b.addEventListener("click", () => { close(); fn(); });
        m.appendChild(b);
        return b;
    };
    const close = () => {
        m.remove();
        document.removeEventListener("pointerdown", outside, true);
        document.removeEventListener("keydown", esc, true);
    };
    const outside = (ev) => { if (!m.contains(ev.target)) close(); };
    const esc = (ev) => { if (ev.key === "Escape") { close(); document.getElementById(PILL_ID)?.focus(); } };
    const first = item("Reset position", resetPosition);
    item("Hide the button (Settings › C2C › Workflow issues)", () => setShow(false));
    document.body.appendChild(m);
    const r = m.getBoundingClientRect();
    const box = clampRect({ x, y, w: r.width, h: r.height }, viewport());
    m.style.left = `${box.x}px`;
    m.style.top = `${box.y}px`;
    setTimeout(() => {
        document.addEventListener("pointerdown", outside, true);
        document.addEventListener("keydown", esc, true);
    }, 0);
    first.focus();
}

function resetPosition() {
    saveState({ pos: null, panel: null });
    placePill();
}

function setShow(v) {
    setS(S_SHOW, !!v);
    _render();
}

// ── the list panel ──────────────────────────────────────────────────────────
function fillList(l) {
    const body = l.querySelector("[data-role=body]");
    const title = l.querySelector("[data-role=title]");
    if (title) title.textContent = `Workflow issues (${_issues.length})`;
    body.replaceChildren();
    for (const iss of _issues) {
        const row = document.createElement("div");
        row.style.cssText = "display:flex;gap:8px;align-items:center;padding:6px;border-bottom:1px solid var(--c2c-surface0);";
        const txt = document.createElement("div");
        txt.style.cssText = "flex:1;line-height:1.4;";
        txt.textContent = iss.msg;
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = iss.fix;
        btn.style.cssText =
            "flex:0 0 auto;padding:3px 10px;border-radius:6px;cursor:pointer;border:none;" +
            "background:var(--c2c-blue);color:var(--c2c-onAccent);font:600 10px ui-sans-serif;";
        btn.addEventListener("click", () => { try { iss.apply(); } catch (_) {} setTimeout(_scan, 250); });
        row.append(txt, btn);
        body.appendChild(row);
    }
    if (!_issues.length) {
        const ok = document.createElement("div");
        ok.style.padding = "14px";
        ok.textContent = "No issues — workflow looks healthy.";
        body.appendChild(ok);
    }
}

function _toggleList() {
    const open = document.getElementById(LIST_ID);
    if (open) { open.remove(); return; }
    const l = document.createElement("div");
    l.id = LIST_ID;
    l.setAttribute("role", "dialog");
    l.setAttribute("aria-label", "Workflow issues");
    l.tabIndex = -1;
    l.style.cssText =
        `position:fixed;left:0;top:0;z-index:${Z + 1};width:380px;height:auto;max-height:70vh;min-width:240px;` +
        "min-height:90px;display:flex;flex-direction:column;overflow:hidden;resize:both;" +
        "background:var(--c2c-bg);border:1px solid var(--c2c-surface2);border-radius:10px;" +
        "box-shadow:0 8px 28px rgba(0,0,0,.5);font:11px ui-sans-serif,system-ui;color:var(--c2c-fg);";
    const head = document.createElement("div");
    head.style.cssText = "display:flex;align-items:center;gap:8px;padding:6px 8px 6px 12px;cursor:grab;touch-action:none;"
        + "border-bottom:1px solid var(--c2c-surface1);user-select:none;";
    const title = document.createElement("div");
    title.dataset.role = "title";
    title.style.cssText = "flex:1;font-weight:600;";
    const x = document.createElement("button");
    x.type = "button";
    x.textContent = "×";
    x.setAttribute("aria-label", "Close");
    x.style.cssText = "border:0;background:transparent;color:inherit;font:16px/1 ui-sans-serif;cursor:pointer;padding:0 4px;";
    x.addEventListener("click", () => { l.remove(); document.getElementById(PILL_ID)?.focus(); });
    head.append(title, x);
    const body = document.createElement("div");
    body.dataset.role = "body";
    body.style.cssText = "flex:1;overflow:auto;padding:4px 8px 8px;";
    l.append(head, body);
    l.addEventListener("keydown", (e) => {
        if (e.key === "Escape") { e.stopPropagation(); l.remove(); document.getElementById(PILL_ID)?.focus(); }
    });
    fillList(l);
    document.body.appendChild(l);
    placeList(l);
    dragPanel(l, head);
    watchSize(l);
    (l.querySelector("[data-role=body] button") || l).focus();
}

/** Saved spot and size, else next to the button: above it when the button is in the lower half. */
function placeList(l) {
    const vp = viewport();
    const saved = state().panel;
    if (saved && Number.isFinite(saved.w) && Number.isFinite(saved.h)) {
        l.style.width = `${Math.max(240, Math.min(saved.w, vp.w - 16))}px`;
        l.style.height = `${Math.max(90, Math.min(saved.h, vp.h - 80))}px`;
    }
    const r = l.getBoundingClientRect();
    let box = fromFraction(saved, r.width, r.height, vp);
    if (!box) {
        const p = document.getElementById(PILL_ID)?.getBoundingClientRect();
        const lower = p ? p.top > vp.h / 2 : true;
        const px = p ? p.left : 74, py = p ? (lower ? p.top - r.height - 8 : p.bottom + 8) : vp.h - r.height - 60;
        box = { x: px, y: py, w: r.width, h: r.height };
    }
    box = clampRect(box, vp);
    l.style.left = `${Math.round(box.x)}px`;
    l.style.top = `${Math.round(box.y)}px`;
}

function savePanel(l) {
    const r = l.getBoundingClientRect();
    const vp = viewport();
    saveState({ panel: { ...toFraction({ x: r.left, y: r.top, w: r.width, h: r.height }, vp), w: Math.round(r.width), h: Math.round(r.height) } });
}

function dragPanel(l, head) {
    let d = null;
    head.addEventListener("pointerdown", (e) => {
        if (e.button !== 0 || e.target.closest("button")) return;
        const r = l.getBoundingClientRect();
        d = { id: e.pointerId, sx: e.clientX, sy: e.clientY, ox: r.left, oy: r.top, w: r.width, h: r.height };
        try { head.setPointerCapture(e.pointerId); } catch (_e) { /* synthetic */ }
        head.style.cursor = "grabbing";
    });
    head.addEventListener("pointermove", (e) => {
        if (!d || e.pointerId !== d.id) return;
        const box = clampRect({ x: d.ox + e.clientX - d.sx, y: d.oy + e.clientY - d.sy, w: d.w, h: d.h }, viewport());
        l.style.left = `${Math.round(box.x)}px`;
        l.style.top = `${Math.round(box.y)}px`;
    });
    const end = (e) => {
        if (!d || e.pointerId !== d.id) return;
        try { head.releasePointerCapture(e.pointerId); } catch (_e) { /* not captured */ }
        d = null;
        head.style.cursor = "grab";
        savePanel(l);
    };
    head.addEventListener("pointerup", end);
    head.addEventListener("pointercancel", end);
}

function watchSize(l) {
    if (typeof ResizeObserver !== "function") return;
    let t = 0, first = true;
    const ro = new ResizeObserver(() => {
        if (first) { first = false; return; }               // the initial report is not a user resize
        clearTimeout(t);
        t = setTimeout(() => { if (l.isConnected) savePanel(l); else ro.disconnect(); }, 400);
    });
    ro.observe(l);
}

// ── render + wiring ─────────────────────────────────────────────────────────
function _render() {
    const p = _pill();
    const list = document.getElementById(LIST_ID);
    if (_issues.length && getS(S_SHOW, true) !== false) {
        const wasHidden = p.style.display === "none";
        p.style.display = "block";
        label(p);
        if (wasHidden) placePill();
    } else {
        p.style.display = "none";
        if (!_issues.length) list?.remove();
    }
    if (list?.isConnected) fillList(list);
}

function openList() {
    _scan().catch(() => {}).finally(() => { if (!document.getElementById(LIST_ID)) _toggleList(); });
}

let _consoleRO = null;
let _watched = new WeakSet();

function watchConsole() {
    if (typeof ResizeObserver !== "function") return;
    _consoleRO ||= new ResizeObserver(() => placePill());
    for (const el of document.querySelectorAll(".bottom-panel")) {
        if (_watched.has(el)) continue;
        _watched.add(el);
        _consoleRO.observe(el);
    }
}

const _arm = () => {
    try {
        if (typeof app?.api?.addEventListener === "function") {
            app.api.addEventListener("graphChanged", _schedule);
        }
    } catch (_) {}
    // Safety-net poll: cheap (object_info cached; scan is O(nodes)).
    getRuntime().every("wizard.scan", 5000, () => _schedule(), { ambient: true });
    // Layout: a few bounding rectangles, never a graph walk, and only while the button is up. Not "ambient": the
    // runtime pauses ambient timers in Lite mode and during runs, and the button must leave the console then too.
    getRuntime().every("issues.layout", 600, () => {
        const p = document.getElementById(PILL_ID);
        if (!p || p.style.display === "none") return;
        watchConsole();
        placePill();
    }, { ambient: false });
    watchConsole();
    window.addEventListener("resize", () => {
        placePill();
        const l = document.getElementById(LIST_ID);
        if (l) placeList(l);
    }, { passive: true });
    _schedule();
};

if (!globalThis.__c2cWorkflowIssuesExt) {
    globalThis.__c2cWorkflowIssuesExt = true;
    app.registerExtension({
        name: "C2C.WorkflowIssues",
        settings: [
            {
                id: S_SHOW,
                name: "Show the Workflow Issues button",
                tooltip: "The '⚠ N workflow issues' button appears while the graph has stale model names or unconnected "
                    + "required inputs. Drag it anywhere; right-click it to reset or hide it. Off = no button (the list "
                    + "still opens from the command 'C2C: Open workflow issues').",
                type: "boolean",
                defaultValue: true,
                category: ["c2c", "Workflow issues", "Button"],
                onChange: () => { try { _render(); } catch (err) { __c2cReport("c2c_workflow_issues", err); } },
            },
            {
                id: S_SIZE,
                name: "Workflow Issues button size",
                type: "combo",
                options: ["small", "normal", "large"],
                defaultValue: "normal",
                category: ["c2c", "Workflow issues", "Size"],
                onChange: () => {
                    const p = document.getElementById(PILL_ID);
                    if (p) { styleP(p); placePill(); }
                },
            },
            { id: S_STATE, name: "Workflow Issues button position", type: "hidden", defaultValue: {} },
        ],
        commands: [
            { id: CMD.SHOW, label: "C2C: Show the Workflow Issues button", function: () => setShow(true) },
            { id: CMD.HIDE, label: "C2C: Hide the Workflow Issues button", function: () => setShow(false) },
            { id: CMD.RESET, label: "C2C: Reset the Workflow Issues button position", function: resetPosition },
            { id: CMD.OPEN, label: "C2C: Open workflow issues", function: openList },
        ],
        setup() {
            try { _arm(); } catch (err) { __c2cReport("c2c_workflow_issues", err); }
        },
        // A loaded workflow is checked at once: loading fires no graphChanged, and the safety poll is ambient (paused
        // in Lite mode), so a broken workflow used to show nothing until the first edit.
        afterConfigureGraph() { _schedule(); },
    });
}
