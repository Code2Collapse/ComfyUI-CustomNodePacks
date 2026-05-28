// c2c_registry_status.js — surface optional-component failures to the user.
//
// Per ideas_summary.md §2.1: the #1 reason the pack feels like stubs is
// that optional sub-imports were silently swallowed. The Python side now
// records every miss to /c2c/registry/status with a `hint` field. This
// extension queries that endpoint shortly after boot and:
//
//   1. Prints a single grouped summary line to the browser console (visible
//      to anyone who opens devtools, which power users do reflexively).
//   2. Pops a single non-blocking notice in the lower-right corner if
//      anything failed (dismissible, never auto-pops more than once per
//      ComfyUI session).
//   3. Listens for live `c2c.registry` PromptServer events so any
//      mid-session record_failure() call also surfaces immediately.
//   4. Registers a Settings entry `c2c.registry.showOnBoot` (default ON)
//      so the user can suppress the auto-pop after acknowledging once.
//   5. Adds a command palette entry "C2C: Show component status" that
//      re-opens the notice on demand.
//
// License: Apache-2.0

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
// C (snapshot palette) intentionally NOT imported — all palette colors come
// from live var(--c2c-X) CSS custom properties so they flip on setVariant().
import { T, reducedMotion, z } from "./_c2c_theme.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";

const SETTING_ID  = "c2c.registry.showOnBoot";
const SESSION_KEY = "c2c.registry.suppressedThisSession";
const STYLE_ID    = "c2c-registry-style";
const ROOT_ID     = "c2c-registry-notice";

function injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    // NOTE: palette colors MUST use var(--c2c-X) tokens, not ${C.X} interpolations.
    // C is a snapshot of the active palette at module-load time and does NOT flip
    // when setVariant() is called. CSS custom properties on :root do flip live.
    // Per-variant–insensitive design tokens (T.*, z.*) stay as interpolations.
    s.textContent = `
#${ROOT_ID} {
    position: fixed;
    right: var(--c2c-pad-lg, 12px);
    bottom: var(--c2c-pad-lg, 12px);
    z-index: ${z.toast};
    width: min(420px, 92vw);
    background: var(--c2c-bg);
    color: var(--c2c-fg);
    border: 1px solid var(--c2c-border);
    border-left: 3px solid var(--c2c-yellow);
    border-radius: ${T.radius.lg};
    box-shadow: 0 18px 48px color-mix(in srgb, var(--c2c-shadowBase) 55%, transparent);
    font: 12px/1.5 ui-sans-serif, system-ui, "Segoe UI", sans-serif;
    padding: ${T.pad.lg};
    backdrop-filter: blur(6px);
    transform: translateY(8px);
    opacity: 0;
    transition: opacity var(--c2c-dur-base, 180ms) var(--c2c-ease-out, ease),
                transform var(--c2c-dur-base, 180ms) var(--c2c-ease-out, ease);
}
#${ROOT_ID}.is-open { opacity: 1; transform: translateY(0); }
#${ROOT_ID} .c2c-reg-head {
    display: flex; align-items: center; gap: ${T.gap.md};
    margin-bottom: ${T.pad.sm};
}
#${ROOT_ID} .c2c-reg-title {
    font-weight: 600; color: var(--c2c-yellow); font-size: 13px;
}
#${ROOT_ID} .c2c-reg-close {
    margin-left: auto; background: transparent; border: none;
    color: var(--c2c-dim); cursor: pointer; font-size: 16px; line-height: 1;
    padding: 2px 6px; border-radius: ${T.radius.sm};
}
#${ROOT_ID} .c2c-reg-close:hover { color: var(--c2c-fg); background: var(--c2c-bg2); }
#${ROOT_ID} ul.c2c-reg-list {
    margin: 0; padding: 0; list-style: none; max-height: 38vh; overflow-y: auto;
}
#${ROOT_ID} li.c2c-reg-item {
    padding: ${T.pad.sm} 0; border-top: 1px dashed var(--c2c-border);
}
#${ROOT_ID} li.c2c-reg-item:first-child { border-top: none; }
#${ROOT_ID} .c2c-reg-key { color: var(--c2c-peach); font-weight: 600; }
#${ROOT_ID} .c2c-reg-grp { color: var(--c2c-dim); font-size: 11px; margin-left: 4px; }
#${ROOT_ID} .c2c-reg-msg { color: var(--c2c-sub); margin-top: 2px; word-break: break-word; }
#${ROOT_ID} .c2c-reg-hint {
    color: var(--c2c-green); margin-top: 4px;
    padding: ${T.pad.xs} ${T.pad.sm};
    background: var(--c2c-bg2); border-radius: ${T.radius.sm};
}
#${ROOT_ID} .c2c-reg-foot {
    margin-top: ${T.pad.md}; display: flex; gap: ${T.gap.sm};
    align-items: center; color: var(--c2c-dim); font-size: 11px;
}
#${ROOT_ID} .c2c-reg-foot button {
    margin-left: auto; background: var(--c2c-bg2); color: var(--c2c-fg);
    border: 1px solid var(--c2c-border); border-radius: ${T.radius.sm};
    padding: 4px 10px; cursor: pointer; font: inherit;
}
#${ROOT_ID} .c2c-reg-foot button:hover { background: var(--c2c-bg3); border-color: var(--c2c-mauve); }
`;
    document.head.appendChild(s);
}

function escapeHtml(s) {
    return String(s ?? "")
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function buildNotice(summary) {
    injectStyle();
    let root = document.getElementById(ROOT_ID);
    if (root) root.remove();
    root = document.createElement("div");
    root.id = ROOT_ID;
    const failures = Array.isArray(summary?.failures) ? summary.failures : [];
    const n = failures.length;
    const head = document.createElement("div");
    head.className = "c2c-reg-head";
    head.innerHTML = `
        <span class="c2c-reg-title">⚠ ${n} optional component${n === 1 ? "" : "s"} unavailable</span>
        <button class="c2c-reg-close" title="Dismiss">×</button>
    `;
    root.appendChild(head);

    const list = document.createElement("ul");
    list.className = "c2c-reg-list";
    for (const f of failures) {
        const li = document.createElement("li");
        li.className = "c2c-reg-item";
        li.innerHTML = `
            <div>
                <span class="c2c-reg-key">${escapeHtml(f.key)}</span>
                <span class="c2c-reg-grp">[${escapeHtml(f.group || "root")}]</span>
            </div>
            <div class="c2c-reg-msg"><b>${escapeHtml(f.exception_type)}</b>: ${escapeHtml(f.message)}</div>
            ${f.hint ? `<div class="c2c-reg-hint">→ ${escapeHtml(f.hint)}</div>` : ""}
        `;
        list.appendChild(li);
    }
    root.appendChild(list);

    const foot = document.createElement("div");
    foot.className = "c2c-reg-foot";
    foot.innerHTML = `
        <span>These are <i>optional</i> backends — core nodes still work.</span>
        <button class="c2c-reg-dismiss">Don't show this session</button>
    `;
    root.appendChild(foot);

    document.body.appendChild(root);
    // Animate in (skip if reduced motion).
    if (reducedMotion()) {
        root.classList.add("is-open");
        root.style.transition = "none";
    } else {
        requestAnimationFrame(() => root.classList.add("is-open"));
    }

    root.querySelector(".c2c-reg-close").addEventListener("click", () => closeNotice());
    root.querySelector(".c2c-reg-dismiss").addEventListener("click", () => {
        try { sessionStorage.setItem(SESSION_KEY, "1"); } catch (__c2cErr) { __c2cReport("c2c_registry_status", __c2cErr); }
        closeNotice();
    });
    return root;
}

function closeNotice() {
    const root = document.getElementById(ROOT_ID);
    if (!root) return;
    root.classList.remove("is-open");
    setTimeout(() => root.remove(), reducedMotion() ? 0 : 220);
}

async function fetchSummary() {
    try {
        const r = await api.fetchApi("/c2c/registry/status");
        if (!r.ok) return null;
        return await r.json();
    } catch (_) {
        return null;
    }
}

async function showIfAny(force = false) {
    const summary = await fetchSummary();
    if (!summary) return null;
    const failures = summary.failures || [];
    if (!failures.length) {
        if (force) {
            // Show "all clear" via console only; no toast for an empty list.
            console.log("[C2C registry] all optional components loaded.");
        }
        return summary;
    }
    if (!force) {
        // Boot-path suppression checks.
        const setOn = app?.ui?.settings?.getSettingValue?.(SETTING_ID);
        if (setOn === false) return summary;
        try { if (sessionStorage.getItem(SESSION_KEY) === "1") return summary; } catch (__c2cErr) { __c2cReport("c2c_registry_status", __c2cErr); }
    }
    // Grouped console summary line (always).
    console.warn(
        "[C2C registry] %d optional component(s) failed to load:",
        failures.length,
        failures.map(f => `${f.group}/${f.key} (${f.exception_type})`),
    );
    buildNotice(summary);
    return summary;
}

app.registerExtension({
    name: "C2C.RegistryStatus",
    settings: [
        {
            id: SETTING_ID,
            name: "C2C → Show component-status notice on boot",
            tooltip: "Pop a non-blocking notice in the bottom-right when any optional C2C backend failed to load.",
            type: "boolean",
            defaultValue: true,
        },
    ],
    commands: [
        {
            id: "C2C.registry.show",
            label: "C2C: Show component status",
            function: () => showIfAny(true),
        },
    ],
    async setup() {
        // Live updates: any mid-session record_failure() refreshes the toast.
        // Local debounce handle (was previously stored on `setup._t` which is a
        // ReferenceError inside the object-method body — `setup` is the
        // property key, not an in-scope identifier).
        let _debounceTimer = null;
        try {
            api.addEventListener("c2c.registry", () => {
                // Debounce: rebuild at most once per 500ms.
                if (_debounceTimer) clearTimeout(_debounceTimer);
                _debounceTimer = setTimeout(() => showIfAny(true), 500);
            });
        } catch (__c2cErr) { __c2cReport("c2c_registry_status", __c2cErr); }
        // Initial check, slightly delayed so the rest of boot has time to
        // populate the registry.
        setTimeout(() => { showIfAny(false); }, 2000);
    },
});
