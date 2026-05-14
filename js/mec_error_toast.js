/**
 * mec_error_toast.js — Phase 2: Error Toast Plain-English Translator
 *
 * Observes ComfyUI's PrimeVue toast container, detects red/error toasts,
 * POSTs the message to /mec/translate_error, and replaces the toast text
 * with a friendly explanation. Hovering shows the original message.
 *
 * Setting:
 *   mec.error_toast.enabled — bool, default true
 */

import { app } from "../../scripts/app.js";

const STYLE_ID = "mec-error-toast-style";
const ATTR_DONE = "data-mec-translated";

const _CACHE = new Map();  // Map<message, data>
const _PENDING = new Set();

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.mec-toast-friendly {
    margin-top: 4px;
}
.mec-toast-friendly .mec-toast-cause {
    font-size: 12px;
    line-height: 1.4;
    margin-bottom: 4px;
}
.mec-toast-friendly .mec-toast-fixes {
    margin: 4px 0 0 0;
    padding-left: 16px;
    font-size: 12px;
    line-height: 1.4;
}
.mec-toast-friendly .mec-toast-fixes li {
    margin-bottom: 2px;
}
.mec-toast-friendly .mec-toast-meta {
    margin-top: 6px;
    font-size: 10px;
    color: #6c7086;
    display: flex;
    align-items: center;
    gap: 6px;
}
.mec-toast-friendly .mec-toast-badge {
    padding: 1px 6px;
    border-radius: 10px;
    font-weight: 700;
    background: #313244;
    color: #f9e2af;
}
.mec-toast-friendly .mec-toast-original {
    margin-top: 6px;
    padding: 4px 6px;
    border-left: 2px solid #45475a;
    color: #6c7086;
    font-family: monospace;
    font-size: 10px;
    max-height: 60px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
    display: none;
}
.mec-toast-friendly.show-original .mec-toast-original { display: block; }
.mec-toast-toggle {
    cursor: pointer;
    text-decoration: underline;
    color: #89b4fa;
}
.mec-toast-loading {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: #6c7086;
    margin-top: 4px;
}
.mec-toast-spinner {
    width: 10px;
    height: 10px;
    border: 2px solid #45475a;
    border-top-color: #89b4fa;
    border-radius: 50%;
    animation: mec-spin 0.7s linear infinite;
}
    `.trim();
    document.head.appendChild(style);
}

function _esc(s) {
    return String(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

function _findToastTextNode(toastEl) {
    // PrimeVue toast structure: .p-toast-message-text contains
    // .p-toast-summary and .p-toast-detail (the body).
    return (
        toastEl.querySelector(".p-toast-detail") ||
        toastEl.querySelector(".p-toast-message-text") ||
        toastEl
    );
}

function _extractMessage(toastEl) {
    const summary = toastEl.querySelector(".p-toast-summary")?.textContent || "";
    const detail  = toastEl.querySelector(".p-toast-detail")?.textContent  || "";
    return (summary + "\n" + detail).trim();
}

function _isErrorToast(toastEl) {
    // ComfyUI uses severity="error" → class .p-toast-message-error
    if (toastEl.classList.contains("p-toast-message-error")) return true;
    // Fall back to checking for any "error" hint in class list
    for (const cls of toastEl.classList) {
        if (/error|danger|fail/i.test(cls)) return true;
    }
    return false;
}

function _renderFriendly(textNode, originalMessage, data) {
    const tier = data.tier ?? 1;
    const tierLabel = tier >= 3 ? "☁ Cloud LLM"
                    : tier === 2 ? "🤖 Local LLM"
                    : "📋 Pattern";

    const cause = data.cause || data.headline || "";
    const fixes = Array.isArray(data.fixes) ? data.fixes : [];
    const fixHtml = fixes.length
        ? `<ul class="mec-toast-fixes">${
              fixes.slice(0, 4).map(f => `<li>${_esc(f)}</li>`).join("")
          }</ul>`
        : "";

    textNode.innerHTML = `
        <div class="mec-toast-friendly" ${ATTR_DONE}="1">
            <div class="mec-toast-cause"><strong>${_esc(data.headline || "Error")}</strong></div>
            <div class="mec-toast-cause">${_esc(cause)}</div>
            ${fixHtml}
            <div class="mec-toast-meta">
                <span class="mec-toast-badge">${tierLabel}</span>
                <span class="mec-toast-toggle">Show original</span>
            </div>
            <pre class="mec-toast-original">${_esc(originalMessage)}</pre>
        </div>`.trim();

    const root = textNode.querySelector(".mec-toast-friendly");
    const toggle = textNode.querySelector(".mec-toast-toggle");
    if (toggle && root) {
        toggle.addEventListener("click", (ev) => {
            ev.stopPropagation();
            root.classList.toggle("show-original");
            toggle.textContent = root.classList.contains("show-original")
                ? "Hide original"
                : "Show original";
        });
    }
}

function _renderLoading(textNode, originalMessage) {
    textNode.innerHTML = `
        <div class="mec-toast-friendly">
            <div class="mec-toast-cause"><strong>${_esc(
                originalMessage.split("\n")[0].slice(0, 120)
            )}</strong></div>
            <div class="mec-toast-loading">
                <div class="mec-toast-spinner"></div>
                <span>Translating error…</span>
            </div>
        </div>`.trim();
}

async function _translateToast(toastEl) {
    if (toastEl.getAttribute(ATTR_DONE) === "1") return;

    const enabled = (() => {
        try {
            return app.ui.settings.getSettingValue("mec.error_toast.enabled", true);
        } catch { return true; }
    })();
    if (!enabled) return;

    const message = _extractMessage(toastEl);
    if (!message || message.length < 4) return;

    toastEl.setAttribute(ATTR_DONE, "1");
    const textNode = _findToastTextNode(toastEl);
    if (!textNode) return;

    // Keep the toast open longer so the user can read the translation.
    // ComfyUI uses PrimeVue auto-close; we can't easily extend it, but we
    // can swap content immediately so even a brief view is useful.

    // Cache hit
    if (_CACHE.has(message)) {
        _renderFriendly(textNode, message, _CACHE.get(message));
        return;
    }

    if (_PENDING.has(message)) return;
    _PENDING.add(message);

    _renderLoading(textNode, message);

    try {
        const resp = await fetch("/mec/translate_error", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message }),
        });
        const json = await resp.json();
        _PENDING.delete(message);
        if (!json.success || !json.data) {
            // Restore original message on failure
            textNode.textContent = message;
            return;
        }
        _CACHE.set(message, json.data);
        if (document.body.contains(toastEl)) {
            _renderFriendly(textNode, message, json.data);
        }
    } catch (err) {
        _PENDING.delete(message);
        console.warn("[MEC.ErrorToast] translate failed:", err);
        textNode.textContent = message;
    }
}

function _scanContainer(root) {
    if (!root || !root.querySelectorAll) return;
    const toasts = root.querySelectorAll(".p-toast-message, [class*='p-toast-message']");
    toasts.forEach((t) => {
        if (_isErrorToast(t)) _translateToast(t);
    });
}

function _installObserver() {
    const observer = new MutationObserver((mutations) => {
        for (const mut of mutations) {
            for (const node of mut.addedNodes) {
                if (node.nodeType !== 1) continue;  // Element only
                if (node.classList?.contains("p-toast-message") || node.querySelector?.(".p-toast-message")) {
                    _scanContainer(node.parentElement || node);
                } else {
                    // sometimes the wrapper element is added with toast inside
                    _scanContainer(node);
                }
            }
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
    // Initial scan in case toasts already exist
    _scanContainer(document.body);
}

app.registerExtension({
    name: "MEC.ErrorToast",
    settings: [
        {
            id: "mec.error_toast.enabled",
            name: "Error Toast: plain-English rewrite",
            tooltip: "Rewrite red error toasts using the MEC error explainer.",
            type: "boolean",
            defaultValue: true,
        },
    ],
    async setup() {
        _injectStyle();
        _installObserver();
        console.log("[MEC.ErrorToast] Loaded — red error toasts will be rewritten.");
    },
});
