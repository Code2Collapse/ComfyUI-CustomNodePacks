/**
 * _c2c_dialog.js — in-app modal dialogs for the C2C pack.
 *
 * ComfyUI's runtime blocks the native browser dialogs `window.prompt()`,
 * `window.confirm()` and `window.alert()` ("prompt() is not supported."),
 * so any code that called them threw and broke the surrounding UI flow.
 * This module provides themed, promise-based replacements:
 *
 *   import { c2cPrompt, c2cConfirm, c2cAlert } from "./_c2c_dialog.js";
 *
 *   const name = await c2cPrompt("Preset name?", "untitled");   // string | null
 *   if (await c2cConfirm("Delete this preset?")) { ... }        // boolean
 *   await c2cAlert("Saved.");                                   // void
 *
 * Design:
 *   • Theme tokens only (var(--c2c-*)) so it flips across Mocha/Latte/OLED.
 *   • Keyboard: Enter = confirm/OK, Esc = cancel; input auto-focused + selected.
 *   • Click on the backdrop = cancel (same as Esc).
 *   • Focus is restored to the previously-focused element on close.
 *   • Only one dialog at a time; a second call queues behind the first.
 *
 * License: Apache-2.0
 */

import { reportFailure as __c2cReport } from "./_c2c_report.js";

const ROOT_ID = "c2c-dialog-root";
let _queue = Promise.resolve();

function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
}

/**
 * Core renderer. Returns a Promise that resolves with:
 *   • prompt  → entered string, or null if cancelled
 *   • confirm → true / false
 *   • alert   → true
 * @param {object} cfg
 * @param {"prompt"|"confirm"|"alert"} cfg.kind
 * @param {string} cfg.message
 * @param {string} [cfg.defaultValue]
 * @param {string} [cfg.okLabel]
 * @param {string} [cfg.cancelLabel]
 * @param {string} [cfg.title]
 */
function _render(cfg) {
    return new Promise((resolve) => {
        let settled = false;
        const prev = document.activeElement;

        const backdrop = document.createElement("div");
        backdrop.id = ROOT_ID;
        backdrop.setAttribute("role", "dialog");
        backdrop.setAttribute("aria-modal", "true");
        backdrop.style.cssText =
            "position:fixed;inset:0;z-index:2000000;display:flex;align-items:center;" +
            "justify-content:center;background:rgba(0,0,0,0.55);" +
            "font-family:ui-sans-serif,system-ui,sans-serif;";

        const card = document.createElement("div");
        card.style.cssText =
            "min-width:320px;max-width:min(560px,92vw);background:var(--c2c-bg2);" +
            "color:var(--c2c-fg);border:1px solid var(--c2c-border);border-radius:8px;" +
            "box-shadow:0 12px 48px rgba(0,0,0,0.5);padding:18px 18px 14px;box-sizing:border-box;";

        const isPrompt = cfg.kind === "prompt";
        const isAlert = cfg.kind === "alert";
        const okLabel = cfg.okLabel || (isAlert ? "OK" : isPrompt ? "OK" : "Yes");
        const cancelLabel = cfg.cancelLabel || (isPrompt ? "Cancel" : "No");

        card.innerHTML = `
            ${cfg.title ? `<div style="font-weight:600;color:var(--c2c-mauve);font-size:13px;margin-bottom:8px">${esc(cfg.title)}</div>` : ""}
            <div style="font-size:13px;line-height:1.5;white-space:pre-wrap;margin-bottom:14px">${esc(cfg.message)}</div>
            ${isPrompt ? `<input id="c2c-dlg-input" type="text" style="width:100%;box-sizing:border-box;background:var(--c2c-bg);color:var(--c2c-fg);border:1px solid var(--c2c-border);border-radius:5px;padding:7px;font-size:13px;margin-bottom:14px" />` : ""}
            <div style="display:flex;gap:8px;justify-content:flex-end">
                ${isAlert ? "" : `<button id="c2c-dlg-cancel" style="background:var(--c2c-bg);color:var(--c2c-fg);border:1px solid var(--c2c-border);border-radius:5px;padding:7px 14px;cursor:pointer;font-size:12px">${esc(cancelLabel)}</button>`}
                <button id="c2c-dlg-ok" style="background:var(--c2c-mauve);color:var(--c2c-bg);border:none;border-radius:5px;padding:7px 16px;cursor:pointer;font-size:12px;font-weight:600">${esc(okLabel)}</button>
            </div>`;

        backdrop.appendChild(card);
        document.body.appendChild(backdrop);

        const input = card.querySelector("#c2c-dlg-input");
        const okBtn = card.querySelector("#c2c-dlg-ok");
        const cancelBtn = card.querySelector("#c2c-dlg-cancel");

        const cleanup = () => {
            document.removeEventListener("keydown", onKey, true);
            if (backdrop.parentElement) backdrop.parentElement.removeChild(backdrop);
            try { if (prev && typeof prev.focus === "function") prev.focus(); } catch (_e) { /* ignore */ }
        };
        const finish = (val) => {
            if (settled) return;
            settled = true;
            cleanup();
            resolve(val);
        };
        const onOk = () => {
            if (isAlert) finish(true);
            else if (isPrompt) finish(input ? input.value : "");
            else finish(true);
        };
        const onCancel = () => {
            if (isAlert) finish(true);
            else if (isPrompt) finish(null);
            else finish(false);
        };

        const onKey = (e) => {
            if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); onCancel(); }
            else if (e.key === "Enter" && (!isPrompt || document.activeElement === input || document.activeElement === okBtn)) {
                e.preventDefault(); e.stopPropagation(); onOk();
            }
        };

        okBtn.addEventListener("click", onOk);
        if (cancelBtn) cancelBtn.addEventListener("click", onCancel);
        backdrop.addEventListener("mousedown", (e) => { if (e.target === backdrop) onCancel(); });
        document.addEventListener("keydown", onKey, true);

        // Focus management.
        if (input) {
            input.value = cfg.defaultValue != null ? String(cfg.defaultValue) : "";
            requestAnimationFrame(() => { try { input.focus(); input.select(); } catch (_e) { /* ignore */ } });
        } else {
            requestAnimationFrame(() => { try { okBtn.focus(); } catch (_e) { /* ignore */ } });
        }
    });
}

/** Serialise dialogs so two near-simultaneous calls don't stack. */
function _enqueue(cfg, fallback) {
    const run = _queue.then(() => _render(cfg).catch((exc) => {
        __c2cReport("_c2c_dialog", exc);
        return fallback;
    }));
    // Keep the chain alive even if a consumer never awaits.
    _queue = run.then(() => undefined, () => undefined);
    return run;
}

/**
 * Themed replacement for window.prompt(). Resolves with the entered string,
 * or null if the user cancels.
 * @param {string} message
 * @param {string} [defaultValue]
 * @param {object} [opts] { title, okLabel, cancelLabel }
 * @returns {Promise<string|null>}
 */
export function c2cPrompt(message, defaultValue = "", opts = {}) {
    return _enqueue({ kind: "prompt", message, defaultValue, ...opts }, null);
}

/**
 * Themed replacement for window.confirm(). Resolves true/false.
 * @param {string} message
 * @param {object} [opts] { title, okLabel, cancelLabel }
 * @returns {Promise<boolean>}
 */
export function c2cConfirm(message, opts = {}) {
    return _enqueue({ kind: "confirm", message, ...opts }, false);
}

/**
 * Themed replacement for window.alert(). Resolves when dismissed.
 * @param {string} message
 * @param {object} [opts] { title, okLabel }
 * @returns {Promise<void>}
 */
export function c2cAlert(message, opts = {}) {
    return _enqueue({ kind: "alert", message, ...opts }, undefined).then(() => undefined);
}
