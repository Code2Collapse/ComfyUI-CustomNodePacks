/**
 * c2c_error_toast.js — Phase 2: Error Toast Plain-English Translator
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
    color: var(--c2c-overlay0);
    display: flex;
    align-items: center;
    gap: 6px;
}
.mec-toast-friendly .mec-toast-badge {
    padding: 1px 6px;
    border-radius: 10px;
    font-weight: 700;
    background: var(--c2c-surface0);
    color: var(--c2c-yellow);
}
.mec-toast-friendly .mec-toast-original {
    margin-top: 6px;
    padding: 4px 6px;
    border-left: 2px solid var(--c2c-surface1);
    color: var(--c2c-overlay0);
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
    color: var(--c2c-blue);
}
.mec-toast-loading {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: var(--c2c-overlay0);
    margin-top: 4px;
}
.mec-toast-spinner {
    width: 10px;
    height: 10px;
    border: 2px solid var(--c2c-surface1);
    border-top-color: var(--c2c-blue);
    border-radius: 50%;
    animation: mec-spin 0.7s linear infinite;
}
@keyframes mec-spin { to { transform: rotate(360deg); } }
.mec-toast-teach {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 8px; margin-left: 6px;
    border-radius: 10px;
    background: var(--c2c-surface0); color: var(--c2c-yellow);
    border: 1px solid var(--c2c-surface1);
    font-size: 10px; font-weight: 700; cursor: pointer;
}
.mec-toast-teach:hover { background: var(--c2c-surface1); color: var(--c2c-peach); }
.mec-teach-mask {
    position: fixed; inset: 0; z-index: var(--c2c-z-toast, 100002);
    background: rgba(0,0,0,0.55);
    display: flex; align-items: center; justify-content: center;
    font-family: var(--p-font-family, system-ui), sans-serif;
}
.mec-teach-dlg {
    width: min(560px, 92vw); max-height: 86vh;
    background: var(--c2c-bg); color: var(--c2c-fg);
    border: 1px solid var(--c2c-surface1); border-radius: 8px;
    box-shadow: 0 12px 48px rgba(0,0,0,0.5);
    display: flex; flex-direction: column; overflow: hidden;
    font-size: 12px;
}
.mec-teach-dlg header {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 14px;
    background: var(--c2c-bg2); border-bottom: 1px solid var(--c2c-surface0);
    font-weight: 600; font-size: 13px;
}
.mec-teach-dlg header .sp { flex: 1; }
.mec-teach-dlg header button {
    background: transparent; border: none; color: inherit;
    font-size: 18px; cursor: pointer; padding: 0 4px;
}
.mec-teach-dlg .body { padding: 12px 14px; overflow: auto; }
.mec-teach-dlg label {
    display: block; font-size: 11px;
    opacity: 0.75; margin: 8px 0 4px; font-weight: 600;
    letter-spacing: 0.4px; text-transform: uppercase;
}
.mec-teach-dlg input[type=text],
.mec-teach-dlg textarea {
    width: 100%; box-sizing: border-box;
    background: var(--c2c-bg3); color: var(--c2c-fg);
    border: 1px solid var(--c2c-surface0); border-radius: 4px;
    padding: 6px 8px; font-size: 12px; font-family: inherit;
}
.mec-teach-dlg textarea { min-height: 60px; resize: vertical;
    font-family: ui-monospace, "Cascadia Mono", monospace; font-size: 11px; }
.mec-teach-dlg .fixrow {
    display: flex; gap: 4px; margin-bottom: 4px;
}
.mec-teach-dlg .fixrow input { flex: 1; }
.mec-teach-dlg .fixrow button {
    background: var(--c2c-surface0); color: var(--c2c-red); border: 1px solid var(--c2c-surface1);
    border-radius: 4px; padding: 2px 8px; cursor: pointer; font-size: 11px;
}
.mec-teach-dlg .addfix {
    background: var(--c2c-surface0); color: var(--c2c-okSoft); border: 1px solid var(--c2c-surface1);
    border-radius: 4px; padding: 3px 10px; cursor: pointer; font-size: 11px;
    margin-top: 4px;
}
.mec-teach-dlg footer {
    display: flex; gap: 8px; align-items: center;
    padding: 10px 14px; border-top: 1px solid var(--c2c-surface0);
    background: var(--c2c-bg2);
}
.mec-teach-dlg footer .sp { flex: 1; }
.mec-teach-dlg footer button {
    padding: 5px 14px; border-radius: 4px; cursor: pointer;
    font-size: 12px; font-weight: 600; border: 1px solid var(--c2c-surface1);
}
.mec-teach-dlg .ok-btn { background: var(--c2c-okSoft); color: var(--c2c-bg3); border-color: var(--c2c-okSoft); }
.mec-teach-dlg .ok-btn:hover { filter: brightness(1.1); }
.mec-teach-dlg .cancel-btn { background: var(--c2c-surface0); color: var(--c2c-fg); }
.mec-teach-dlg .cancel-btn:hover { background: var(--c2c-surface1); }
.mec-teach-dlg .err { color: var(--c2c-red); font-size: 11px; padding: 4px 0; }
.mec-teach-dlg .raw-preview {
    background: var(--c2c-bg3); border-left: 2px solid var(--c2c-surface1);
    padding: 6px 8px; margin: 4px 0 8px;
    font-family: ui-monospace, "Cascadia Mono", monospace;
    font-size: 10px; color: var(--c2c-overlay0);
    max-height: 70px; overflow: auto; white-space: pre-wrap;
    word-break: break-word; border-radius: 0 4px 4px 0;
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

    const noMatch = (data.pattern_id === "no_match" || data.tier1_match === "no_match"
                    || data.tier1_match == null && tier >= 2);
    const teachBtn = noMatch
        ? `<button class="mec-toast-teach" data-role="teach"
                   title="Teach MEC how to recognise this error next time">📚 Teach me</button>`
        : "";

    textNode.innerHTML = `
        <div class="mec-toast-friendly" ${ATTR_DONE}="1">
            <div class="mec-toast-cause"><strong>${_esc(data.headline || "Error")}</strong></div>
            <div class="mec-toast-cause">${_esc(cause)}</div>
            ${fixHtml}
            <div class="mec-toast-meta">
                <span class="mec-toast-badge">${tierLabel}</span>
                <span class="mec-toast-toggle">Show original</span>
                ${teachBtn}
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
    const teach = textNode.querySelector('[data-role="teach"]');
    if (teach) {
        teach.addEventListener("click", (ev) => {
            ev.stopPropagation();
            _openTeachDialog(originalMessage, data);
        });
    }
}

// ── "Teach me" dialog ───────────────────────────────────────────────
let _TEACH_DLG = null;
function _closeTeachDialog() {
    if (_TEACH_DLG) { _TEACH_DLG.remove(); _TEACH_DLG = null; }
}
function _openTeachDialog(originalMessage, data) {
    _closeTeachDialog();
    const firstLine = (originalMessage || "").split("\n")[0].slice(0, 200);
    // Suggest a forgiving regex client-side; backend will re-derive if blank.
    const suggestedRegex = firstLine
        .replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
        .replace(/\d+/g, "\\d+")
        .replace(/\s+/g, "\\s+");
    const tier = data.tier ?? 0;

    const mask = document.createElement("div");
    mask.className = "mec-teach-mask";
    mask.addEventListener("click", (e) => { if (e.target === mask) _closeTeachDialog(); });

    const dlg = document.createElement("div");
    dlg.className = "mec-teach-dlg";
    dlg.innerHTML = `
        <header>
            <span>📚 Teach MEC about this error</span>
            <span class="sp"></span>
            <button data-role="x" title="Close">×</button>
        </header>
        <div class="body">
            <label>Original error</label>
            <div class="raw-preview">${_esc(originalMessage).slice(0, 800)}</div>
            <label>Pattern ID (optional, lowercase slug)</label>
            <input type="text" data-role="id" placeholder="(auto from message)">
            <label>Match regex (Python re, case-insensitive)</label>
            <input type="text" data-role="regex" value="${_esc(suggestedRegex)}">
            <label>Cause — what does this mean? *</label>
            <textarea data-role="cause" placeholder="${_esc(data?.cause || '')}">${_esc(data?.cause || '')}</textarea>
            <label>Fixes — one per row</label>
            <div data-role="fixes"></div>
            <button class="addfix" data-role="addfix">＋ Add fix</button>
            <label>Category</label>
            <input type="text" data-role="cat" value="user">
            <div class="err" data-role="err" style="display:none"></div>
        </div>
        <footer>
            <span style="font-size:10px;opacity:0.6;">Saved to <code>patterns/user/learned.json</code></span>
            <span class="sp"></span>
            <button class="cancel-btn" data-role="cancel">Cancel</button>
            <button class="ok-btn" data-role="save">Save pattern</button>
        </footer>`;

    mask.appendChild(dlg);
    document.body.appendChild(mask);
    _TEACH_DLG = mask;

    const fixesWrap = dlg.querySelector('[data-role="fixes"]');
    const addFix = (val = "") => {
        const row = document.createElement("div");
        row.className = "fixrow";
        row.innerHTML = `<input type="text" value="${_esc(val)}" placeholder="Suggested fix">
                         <button type="button" title="Remove">−</button>`;
        row.querySelector("button").addEventListener("click", () => row.remove());
        fixesWrap.appendChild(row);
        return row;
    };
    const initialFixes = Array.isArray(data?.fixes) && data.fixes.length ? data.fixes : [""];
    initialFixes.slice(0, 6).forEach(f => addFix(String(f || "")));

    dlg.querySelector('[data-role="addfix"]').addEventListener("click", () => addFix(""));
    dlg.querySelector('[data-role="x"]').addEventListener("click", _closeTeachDialog);
    dlg.querySelector('[data-role="cancel"]').addEventListener("click", _closeTeachDialog);

    dlg.querySelector('[data-role="save"]').addEventListener("click", async () => {
        const errBox = dlg.querySelector('[data-role="err"]');
        errBox.style.display = "none";
        const cause = dlg.querySelector('[data-role="cause"]').value.trim();
        if (!cause) {
            errBox.textContent = "Cause is required.";
            errBox.style.display = "block";
            return;
        }
        const payload = {
            message: originalMessage,
            cause,
            regex:  dlg.querySelector('[data-role="regex"]').value.trim() || undefined,
            id:     dlg.querySelector('[data-role="id"]').value.trim() || undefined,
            category: dlg.querySelector('[data-role="cat"]').value.trim() || "user",
            source_tier: tier || undefined,
            fixes: Array.from(fixesWrap.querySelectorAll("input"))
                        .map(i => i.value.trim()).filter(Boolean),
        };
        try {
            const r = await fetch("/mec/teach_error", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const j = await r.json();
            if (!j.success) {
                errBox.textContent = j.message || "Save failed.";
                errBox.style.display = "block";
                return;
            }
            // Cache the new translation so subsequent identical toasts hit fast path.
            _CACHE.set(originalMessage, {
                tier: 1,
                headline: data?.headline || originalMessage.split("\n")[0],
                cause,
                fixes: payload.fixes,
                pattern_id: j.data.id,
                category: payload.category,
                confidence: 0.7,
            });
            try {
                app.extensionManager?.toast?.add({
                    severity: "success",
                    summary: "MEC learned a new pattern",
                    detail: `Saved as "${j.data.id}". Next time this fires it will match instantly.`,
                    life: 4000,
                });
            } catch {}
            _closeTeachDialog();
        } catch (e) {
            errBox.textContent = String(e);
            errBox.style.display = "block";
        }
    });

    setTimeout(() => dlg.querySelector('[data-role="cause"]').focus(), 0);
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
            return app.ui.settings.getSettingValue("c2c.error_toast.enabled", true);
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
    let rafPending = false;
    let pendingRoots = [];
    const flush = () => {
        rafPending = false;
        const roots = pendingRoots;
        pendingRoots = [];
        for (const r of roots) _scanContainer(r);
    };
    const enqueue = (node) => {
        pendingRoots.push(node);
        if (rafPending) return;
        rafPending = true;
        const raf = (typeof requestAnimationFrame !== "undefined")
            ? requestAnimationFrame
            : (cb) => setTimeout(cb, 16);
        raf(flush);
    };

    // PrimeVue mounts `.p-toast` as a direct child of body (usually exactly
    // one container) and then adds `.p-toast-message` children inside it
    // each time a toast appears. We avoid the costly `subtree:true` on body
    // by:
    //   1. A cheap body-level childList observer that only fires on top-
    //      level mounts → upgrades to per-container observers when found.
    //   2. A scoped observer on each `.p-toast` container with childList
    //      (no subtree) to catch each new message.
    const scopedObservers = new WeakSet();
    const attachToContainer = (container) => {
        if (!container || scopedObservers.has(container)) return;
        scopedObservers.add(container);
        const o = new MutationObserver((muts) => {
            for (const m of muts) {
                for (const n of m.addedNodes) {
                    if (n.nodeType === 1) enqueue(n);
                }
            }
        });
        o.observe(container, { childList: true, subtree: false });
        // Catch any messages already present.
        enqueue(container);
    };

    const findAndAttachContainers = () => {
        document.querySelectorAll(".p-toast").forEach(attachToContainer);
    };

    const bodyObserver = new MutationObserver((mutations) => {
        for (const mut of mutations) {
            for (const node of mut.addedNodes) {
                if (node.nodeType !== 1) continue;
                if (node.classList?.contains("p-toast")) {
                    attachToContainer(node);
                } else if (node.querySelector) {
                    node.querySelectorAll?.(".p-toast")
                        .forEach(attachToContainer);
                }
            }
        }
    });
    bodyObserver.observe(document.body, { childList: true, subtree: false });

    // Initial: existing containers + any pre-existing messages.
    findAndAttachContainers();
    _scanContainer(document.body);
}

app.registerExtension({
    name: "C2C.ErrorToast",
    settings: [
        {
            id: "c2c.error_toast.enabled",
            name: "Error Toast: plain-English rewrite",
            tooltip: "Rewrite red error toasts using the MEC error explainer.",
            type: "boolean",
            default: true,
        },
    ],
    async setup() {
        _injectStyle();
        _installObserver();
        console.log("[MEC.ErrorToast] Loaded — red error toasts will be rewritten.");
    },
});
