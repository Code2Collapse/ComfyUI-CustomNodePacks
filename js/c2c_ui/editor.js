/**
 * c2c_ui/editor.js — full-screen C2C editor shell.
 *
 * License: Apache-2.0
 */

import { ensureStyles } from "./theme.js";
import { wolfMark } from "./wolf.js";
import { button } from "./components.js";

const FOCUSABLE = 'a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])';

/**
 * @param {{title: string, left?: HTMLElement, right?: HTMLElement, centre?: HTMLElement, onSave?: () => void|boolean|Promise<void|boolean>, onClose?: () => void, onUndo?: () => void, onRedo?: () => void, help?: () => void, hints?: string|string[]}} opts
 * @returns {{close: () => void, setDirty: (d: boolean) => void, setStatus: (t: string) => void, el: HTMLElement}}
 */
export function openEditor(opts) {
    ensureStyles();

    let dirty = false;
    let closed = false;
    const prevFocus = document.activeElement;

    const overlay = document.createElement("div");
    overlay.className = "c2c-ui-editor c2c-ui-editor-overlay";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", opts.title || "Editor");
    overlay.tabIndex = -1;   // focus lands on the dialog itself, not on Undo

    // Keep the editor's input away from ComfyUI underneath. BUBBLE phase on
    // purpose: controls inside the editor (a paint canvas, a drag slider, an
    // input) handle the event first, then it stops here before it can reach
    // the graph canvas or ComfyUI's shortcuts. (Stopping in the capture phase
    // at the overlay would starve every control inside it.)
    const blockToGraph = (e) => { e.stopPropagation(); };
    for (const type of ["keydown", "keyup", "keypress", "wheel", "pointerdown", "pointerup", "pointermove", "contextmenu"]) {
        overlay.addEventListener(type, blockToGraph, false);
    }

    // Header
    const header = document.createElement("div");
    header.className = "c2c-ui-editor__header";

    const brand = document.createElement("div");
    brand.className = "c2c-ui-editor__brand";
    brand.appendChild(wolfMark(24));
    const brandText = document.createElement("span");
    brandText.className = "c2c-ui-editor__brand-text";
    brandText.textContent = "C2C";
    brand.appendChild(brandText);

    const title = document.createElement("h2");
    title.className = "c2c-ui-editor__title";
    title.textContent = opts.title || "";

    const headerActions = document.createElement("div");
    headerActions.className = "c2c-ui-editor__header-actions";

    const undoBtn = button("Undo", { disabled: !opts.onUndo, onClick: () => opts.onUndo?.() });
    const redoBtn = button("Redo", { disabled: !opts.onRedo, onClick: () => opts.onRedo?.() });
    const helpBtn = button("Help", { disabled: !opts.help, onClick: () => opts.help?.() });
    headerActions.appendChild(undoBtn);
    headerActions.appendChild(redoBtn);
    headerActions.appendChild(helpBtn);

    header.appendChild(brand);
    header.appendChild(title);
    header.appendChild(headerActions);

    // Body
    const body = document.createElement("div");
    body.className = "c2c-ui-editor__body";

    const left = document.createElement("div");
    left.className = "c2c-ui-editor__left";
    if (opts.left) left.appendChild(opts.left);

    const centre = document.createElement("div");
    centre.className = "c2c-ui-editor__centre";
    if (opts.centre) centre.appendChild(opts.centre);

    const right = document.createElement("div");
    right.className = "c2c-ui-editor__right";
    if (opts.right) right.appendChild(opts.right);

    body.appendChild(left);
    body.appendChild(centre);
    body.appendChild(right);

    // Confirm bar (dirty Esc)
    const confirmBar = document.createElement("div");
    confirmBar.className = "c2c-ui-editor__confirm";
    confirmBar.hidden = true;
    const confirmText = document.createElement("span");
    confirmText.className = "c2c-ui-editor__confirm-text";
    confirmText.textContent = "Discard unsaved changes?";
    const confirmDiscard = button("Discard", { danger: true });
    const confirmStay = button("Keep editing");
    confirmBar.appendChild(confirmText);
    confirmBar.appendChild(confirmDiscard);
    confirmBar.appendChild(confirmStay);

    // Footer
    const footer = document.createElement("div");
    footer.className = "c2c-ui-editor__footer";

    const hints = document.createElement("div");
    hints.className = "c2c-ui-editor__hints";
    const hintList = Array.isArray(opts.hints) ? opts.hints : (opts.hints ? [opts.hints] : []);
    hints.textContent = hintList.join("  \u00B7  ");

    const actions = document.createElement("div");
    actions.className = "c2c-ui-editor__actions";
    const saveBtn = button("Save", { primary: true });
    const closeBtn = button("Close");
    actions.appendChild(saveBtn);
    actions.appendChild(closeBtn);

    footer.appendChild(hints);
    footer.appendChild(actions);

    overlay.appendChild(header);
    overlay.appendChild(body);
    overlay.appendChild(confirmBar);
    overlay.appendChild(footer);
    document.body.appendChild(overlay);

    const getFocusables = () => [...overlay.querySelectorAll(FOCUSABLE)].filter(
        (el) => el.offsetParent !== null && !el.disabled,
    );

    const trapFocus = (e) => {
        if (e.key !== "Tab" || closed) return;
        const items = getFocusables();
        if (!items.length) return;
        const first = items[0];
        const last = items[items.length - 1];
        if (e.shiftKey && document.activeElement === first) {
            e.preventDefault();
            last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
            e.preventDefault();
            first.focus();
        }
    };

    const doClose = () => {
        if (closed) return;
        closed = true;
        document.removeEventListener("keydown", onKey, true);
        overlay.removeEventListener("keydown", trapFocus, true);
        try { opts.onClose?.(); } catch (_e) { /* ignore */ }
        overlay.remove();
        try {
            if (prevFocus && typeof prevFocus.focus === "function") prevFocus.focus();
        } catch (_e) { /* ignore */ }
    };

    const requestClose = () => {
        if (dirty) {
            confirmBar.hidden = false;
            confirmStay.focus();
        } else {
            doClose();
        }
    };

    const onKey = (e) => {
        if (closed) return;

        if (e.key === "Escape") {
            e.preventDefault();
            e.stopPropagation();
            if (!confirmBar.hidden) {
                confirmBar.hidden = true;
            } else {
                requestClose();
            }
            return;
        }

        const mod = e.ctrlKey || e.metaKey;
        // Text fields keep their own undo.
        const t = e.target;
        const typing = t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName || ""));
        if (typing) return;
        if (mod && e.key === "z" && !e.shiftKey && opts.onUndo) {
            e.preventDefault();
            e.stopPropagation();
            opts.onUndo();
            return;
        }
        if (mod && (e.key === "y" || (e.key === "z" && e.shiftKey)) && opts.onRedo) {
            e.preventDefault();
            e.stopPropagation();
            opts.onRedo();
            return;
        }
    };

    document.addEventListener("keydown", onKey, true);
    overlay.addEventListener("keydown", trapFocus, true);

    const setHint = (text, kind) => {
        hints.textContent = text;
        hints.dataset.kind = kind || "";
    };

    saveBtn.addEventListener("click", async () => {
        saveBtn.disabled = true;
        try {
            const result = await opts.onSave?.();
            if (result !== false) {
                dirty = false;
                doClose();
            }
        } catch (e) {
            // Say what happened, in words; the editor stays open with the work.
            setHint(`Could not save: ${e?.message || e}. Your changes are still here.`, "danger");
        } finally {
            saveBtn.disabled = false;
        }
    });

    closeBtn.addEventListener("click", requestClose);
    confirmDiscard.addEventListener("click", () => { dirty = false; doClose(); });
    confirmStay.addEventListener("click", () => { confirmBar.hidden = true; });

    requestAnimationFrame(() => { try { overlay.focus({ preventScroll: true }); } catch (_e) { /* ignore */ } });

    return {
        el: overlay,
        close: doClose,
        setDirty(d) { dirty = !!d; },
        setStatus(text, kind) { setHint(text, kind); },
    };
}
