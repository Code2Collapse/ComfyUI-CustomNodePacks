/**
 * c2c_ui/kit.js — the L4.02 additions to the C2C UI framework: collapsible sections, Simple / Advanced, loading and
 * error states, toasts, one-step undo, pointer drags, and a "?" shortcut overlay. Same tokens (--cu-*) and scope
 * (.c2c-ui) as components.js; works in classic and Nodes 2.0 (plain DOM, no renderer assumptions).
 *
 * License: Apache-2.0
 */
import { ensureStyles } from "./theme.js";

// No import of /scripts/app.js: this folder is vendored into packs at different depths, and one wrong relative
// import kills a pack's whole front-end. The frontend exposes the app on window.comfyAPI.
const _app = () => globalThis.comfyAPI?.app?.app ?? globalThis.app;

const KIT_STYLE_ID = "c2c-ui-kit-style";
const KIT_CSS = `
.c2c-ui details.c2c-ui-collapse { border: 1px solid var(--cu-edge); border-radius: 6px; background: var(--cu-sunken); }
.c2c-ui details.c2c-ui-collapse > summary { cursor: pointer; padding: 4px 8px; font-size: 11px; font-weight: 600;
  color: var(--cu-ink-soft); list-style: none; user-select: none; }
.c2c-ui details.c2c-ui-collapse > summary::-webkit-details-marker { display: none; }
.c2c-ui details.c2c-ui-collapse > summary::before { content: "\\25B8"; display: inline-block; width: 1em;
  transition: transform 120ms; }
.c2c-ui details.c2c-ui-collapse[open] > summary::before { transform: rotate(90deg); }
.c2c-ui details.c2c-ui-collapse > .c2c-ui-collapse__body { padding: 4px 8px 8px; }
.c2c-ui .c2c-ui-mode { display: inline-flex; border: 1px solid var(--cu-edge); border-radius: 6px; overflow: hidden; }
.c2c-ui .c2c-ui-mode > button { font: inherit; font-size: 11px; padding: 2px 10px; border: 0; cursor: pointer;
  background: transparent; color: var(--cu-ink-soft); }
.c2c-ui .c2c-ui-mode > button[aria-pressed="true"] { background: var(--cu-accent); color: var(--cu-on-accent); }
.c2c-ui[data-c2c-mode="simple"] [data-c2c-advanced] { display: none !important; }
.c2c-ui .c2c-ui-state { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 6px;
  padding: 16px; text-align: center; color: var(--cu-ink-soft); font-size: 12px; }
.c2c-ui .c2c-ui-state__spin { width: 18px; height: 18px; border-radius: 50%; border: 2px solid var(--cu-edge);
  border-top-color: var(--cu-accent); animation: c2c-ui-kit-spin 0.8s linear infinite; }
@media (prefers-reduced-motion: reduce) { .c2c-ui .c2c-ui-state__spin { animation-duration: 2.4s; } }
@keyframes c2c-ui-kit-spin { to { transform: rotate(360deg); } }
.c2c-ui .c2c-ui-state--error { color: var(--cu-danger); }
.c2c-ui .c2c-ui-state--error .c2c-ui-state__detail { color: var(--cu-ink-soft); max-width: 36em; }
.c2c-ui-hotkeys { position: fixed; inset: 0; z-index: 10050; display: flex; align-items: center; justify-content: center;
  background: rgba(0, 0, 0, 0.45); }
.c2c-ui-hotkeys__card { background: var(--cu-panel); color: var(--cu-ink); border: 1px solid var(--cu-edge-strong);
  border-radius: 10px; padding: 14px 18px; min-width: 280px; max-width: min(560px, 92vw); max-height: 80vh; overflow: auto; }
.c2c-ui-hotkeys__card h3 { margin: 0 0 8px; font-size: 13px; }
.c2c-ui-hotkeys__grid { display: grid; grid-template-columns: max-content 1fr; gap: 4px 14px; font-size: 12px; }
.c2c-ui-hotkeys__grid kbd { font: 11px ui-monospace, monospace; padding: 1px 6px; border-radius: 4px;
  border: 1px solid var(--cu-edge); background: var(--cu-sunken); white-space: nowrap; }
`;

function ensureKitStyles() {
    ensureStyles();
    if (typeof document === "undefined" || document.getElementById(KIT_STYLE_ID)) return;
    const el = document.createElement("style");
    el.id = KIT_STYLE_ID;
    el.textContent = KIT_CSS;
    document.head.appendChild(el);
}

function _read(key, fallback) {
    try { const v = localStorage.getItem(key); return v == null ? fallback : v; } catch { return fallback; }
}

function _write(key, value) {
    try { localStorage.setItem(key, value); } catch { /* private mode: in-memory only */ }
}

/**
 * Collapsible section. `storageKey` remembers open / closed per browser.
 * @param {string} title
 * @param {{open?: boolean, storageKey?: string}} opts
 * @param  {...(HTMLElement|string)} children
 */
export function collapsible(title, opts = {}, ...children) {
    ensureKitStyles();
    const d = document.createElement("details");
    d.className = "c2c-ui-collapse";
    const open = opts.storageKey ? _read(opts.storageKey, opts.open ? "1" : "0") === "1" : !!opts.open;
    d.open = open;
    const s = document.createElement("summary");
    s.textContent = title;
    s.className = "c2c-ui-focusable";
    const body = document.createElement("div");
    body.className = "c2c-ui-collapse__body";
    for (const c of children) {
        if (typeof c === "string") { const p = document.createElement("p"); p.textContent = c; body.appendChild(p); }
        else if (c) body.appendChild(c);
    }
    d.append(s, body);
    if (opts.storageKey) d.addEventListener("toggle", () => _write(opts.storageKey, d.open ? "1" : "0"));
    return d;
}

/**
 * Simple / Advanced switch. Elements marked `data-c2c-advanced` inside `root` (a `.c2c-ui` element) hide in Simple.
 * @param {HTMLElement} root
 * @param {{value?: "simple"|"advanced", storageKey?: string, onChange?: (mode: string) => void}} opts
 */
export function modeToggle(root, opts = {}) {
    ensureKitStyles();
    let mode = opts.storageKey ? _read(opts.storageKey, opts.value || "simple") : (opts.value || "simple");
    const wrap = document.createElement("div");
    wrap.className = "c2c-ui-mode";
    wrap.setAttribute("role", "group");
    const btns = ["simple", "advanced"].map((m) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "c2c-ui-focusable";
        b.textContent = m === "simple" ? "Simple" : "Advanced";
        b.addEventListener("click", (e) => { e.stopPropagation(); set(m); });
        return b;
    });
    wrap.append(...btns);
    function set(m) {
        mode = m === "advanced" ? "advanced" : "simple";
        root.setAttribute("data-c2c-mode", mode);
        btns[0].setAttribute("aria-pressed", String(mode === "simple"));
        btns[1].setAttribute("aria-pressed", String(mode === "advanced"));
        if (opts.storageKey) _write(opts.storageKey, mode);
        opts.onChange?.(mode);
    }
    set(mode);
    wrap.setMode = set;
    return wrap;
}

/** Loading state: a spinner and one line ("Loading frames..."). */
export function loadingState(label = "Loading...") {
    ensureKitStyles();
    const el = document.createElement("div");
    el.className = "c2c-ui-state";
    el.setAttribute("role", "status");
    const spin = document.createElement("div");
    spin.className = "c2c-ui-state__spin";
    const p = document.createElement("p");
    p.textContent = label;
    p.style.margin = "0";
    el.append(spin, p);
    return el;
}

/**
 * Error state: what went wrong, how to fix it, and an optional retry button.
 * @param {{title: string, detail?: string, retryLabel?: string, onRetry?: () => void}} opts
 */
export function errorState(opts) {
    ensureKitStyles();
    const el = document.createElement("div");
    el.className = "c2c-ui-state c2c-ui-state--error";
    el.setAttribute("role", "alert");
    const t = document.createElement("p");
    t.textContent = opts.title;
    t.style.cssText = "margin:0;font-weight:600";
    el.appendChild(t);
    if (opts.detail) {
        const d = document.createElement("p");
        d.className = "c2c-ui-state__detail";
        d.textContent = opts.detail;
        d.style.margin = "0";
        el.appendChild(d);
    }
    if (opts.onRetry) {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "c2c-ui-btn c2c-ui-focusable";
        b.textContent = opts.retryLabel || "Try again";
        b.addEventListener("click", (e) => { e.stopPropagation(); opts.onRetry(); });
        el.appendChild(b);
    }
    return el;
}

/** ComfyUI's own toast (no second toast system). severity: success | info | warn | error. */
export function toast(summary, detail = "", severity = "info", life = 4000) {
    try {
        _app()?.extensionManager?.toast?.add({ severity, summary, detail, life });
        return true;
    } catch {
        return false;
    }
}

/**
 * Run `fn` as ONE undo step: the frontend's ChangeTracker counts begin / end and records a single state when the
 * outermost step closes (verified against frontend 1.53: canvas.emitBeforeChange -> 'litegraph:canvas'
 * before-change -> changeTracker.beforeChange()). Nested calls collapse into the outermost step.
 */
export function undoStep(fn) {
    const canvas = _app()?.canvas;
    canvas?.emitBeforeChange?.();
    try {
        return fn();
    } finally {
        canvas?.emitAfterChange?.();
    }
}

/**
 * Pointer drag with capture (mouse, pen and touch alike; the drag keeps working outside the element).
 * Returns a function that removes the listeners.
 * @param {HTMLElement} el
 * @param {{onStart?: (e: PointerEvent) => boolean|void, onMove?: (e: PointerEvent, dx: number, dy: number) => void,
 *          onEnd?: (e: PointerEvent, cancelled: boolean) => void, button?: number}} h
 */
export function pointerDrag(el, h = {}) {
    let active = null;
    const down = (e) => {
        if (e.button !== (h.button ?? 0)) return;
        if (h.onStart && h.onStart(e) === false) return;
        active = { id: e.pointerId, x: e.clientX, y: e.clientY };
        try { el.setPointerCapture(e.pointerId); } catch { /* */ }
        e.preventDefault();
        e.stopPropagation();
    };
    const move = (e) => {
        if (!active || e.pointerId !== active.id) return;
        h.onMove?.(e, e.clientX - active.x, e.clientY - active.y);
    };
    const end = (cancelled) => (e) => {
        if (!active || e.pointerId !== active.id) return;
        active = null;
        try { el.releasePointerCapture(e.pointerId); } catch { /* */ }
        h.onEnd?.(e, cancelled);
    };
    const up = end(false), cancel = end(true);
    el.addEventListener("pointerdown", down);
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", cancel);
    return () => {
        el.removeEventListener("pointerdown", down);
        el.removeEventListener("pointermove", move);
        el.removeEventListener("pointerup", up);
        el.removeEventListener("pointercancel", cancel);
    };
}

/**
 * "?" opens a shortcut overlay while focus is inside `scope` (never a page-wide key: the canvas keeps its own keys).
 * Esc or a click outside closes it. Returns a function that removes the binding.
 * @param {HTMLElement} scope
 * @param {string} title
 * @param {Array<[string, string]>} entries  [keys, what it does]
 */
export function hotkeyOverlay(scope, title, entries) {
    ensureKitStyles();
    let overlay = null;
    const close = () => { overlay?.remove(); overlay = null; };
    const open = () => {
        if (overlay) return;
        overlay = document.createElement("div");
        overlay.className = "c2c-ui c2c-ui-hotkeys";
        const card = document.createElement("div");
        card.className = "c2c-ui-hotkeys__card";
        card.setAttribute("role", "dialog");
        card.setAttribute("aria-label", title);
        const h = document.createElement("h3");
        h.textContent = title;
        const grid = document.createElement("div");
        grid.className = "c2c-ui-hotkeys__grid";
        for (const [keys, what] of entries) {
            const k = document.createElement("span");
            for (const part of String(keys).split("+")) {
                const kbd = document.createElement("kbd");
                kbd.textContent = part.trim();
                k.appendChild(kbd);
            }
            const w = document.createElement("span");
            w.textContent = what;
            grid.append(k, w);
        }
        card.append(h, grid);
        overlay.appendChild(card);
        overlay.addEventListener("pointerdown", (e) => { if (e.target === overlay) close(); });
        document.body.appendChild(overlay);
    };
    const onKey = (e) => {
        if (e.key === "Escape" && overlay) { e.stopPropagation(); close(); return; }
        if (e.key !== "?" || e.ctrlKey || e.metaKey || e.altKey) return;
        const t = e.target;
        if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
        if (!scope.contains(t) && !scope.contains(document.activeElement)) return;
        e.preventDefault();
        e.stopPropagation();
        overlay ? close() : open();
    };
    document.addEventListener("keydown", onKey, true);
    return () => { document.removeEventListener("keydown", onKey, true); close(); };
}
