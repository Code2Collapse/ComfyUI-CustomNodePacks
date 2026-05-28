// SPDX-License-Identifier: Apache-2.0
//
// C2C dep-conflict modal — Manager integration.
//
// Watches for calls to ComfyUI-Manager's install endpoints
// (`/manager/queue/install`, `/customnode/install`, `/customnode/install_pip`).
// When one of those calls completes successfully, we wait briefly so the
// Manager has time to finish unpacking the new pack, then ask
// `/c2c/depcheck/scan_new` to report any breaking / risky dependency
// conflicts the new pack introduced against the live environment.
//
// If the server returns breaking entries we pop a modal that:
//   * Lists every conflict in plain English.
//   * Shows the exact `pip install --force-reinstall` command the user
//     can paste into the terminal.
//   * Lets the user copy that command to clipboard with one click.
//
// This is observe-only — we never block the user's install. The modal is
// purely advisory, so we cannot break ComfyUI-Manager itself.

import { app } from "../../scripts/app.js";

const STYLE_TAG_ID = "c2c-depcheck-style";
const MODAL_ID     = "c2c-depcheck-modal";

// Endpoints that Manager calls when the user clicks "Install" / "Update".
// We don't care if there are extras — these three are documented in Manager's
// server.py (v3.x). If any of them appears the post-install scan kicks off.
const INSTALL_ENDPOINTS = [
    "/manager/queue/install",
    "/customnode/install",
    "/customnode/install_pip",
];

// Pip executable to suggest in the remediation command. Best-effort.
const SUGGESTED_PIP_HEAD = `python -m pip install --upgrade --force-reinstall`;

// Cache of the directory list taken at session start. Whatever exists now is
// "old" — anything that shows up later is what Manager just downloaded.
let _baseline = null;

// Prevent re-entrant modals when the user installs several packs back-to-back.
let _modalOpen = false;
// Throttle: don't scan more than once every 4 s.
let _lastScanAt = 0;

function _injectStyles() {
    if (document.getElementById(STYLE_TAG_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_TAG_ID;
    s.textContent = `
    #${MODAL_ID} {
        position: fixed; inset: 0; z-index: var(--c2c-z-modal);
        background: color-mix(in srgb, var(--c2c-shadowBase) 55%, transparent); display: flex;
        align-items: center; justify-content: center;
        font-family: ui-sans-serif, system-ui, sans-serif;
    }
    #${MODAL_ID} .c2c-dc-card {
        background: var(--c2c-bg); color: var(--c2c-fg);
        border: 1px solid var(--c2c-surface1); border-radius: 8px;
        width: min(640px, 92vw); max-height: 85vh;
        display: flex; flex-direction: column;
        box-shadow: 0 10px 30px color-mix(in srgb, var(--c2c-shadowBase) 50%, transparent);
    }
    #${MODAL_ID} .c2c-dc-head {
        padding: 14px 18px; border-bottom: 1px solid var(--c2c-border);
        display: flex; align-items: center; gap: 10px;
    }
    #${MODAL_ID} .c2c-dc-head .icon {
        font-size: 22px; color: var(--c2c-yellow);
    }
    #${MODAL_ID} .c2c-dc-head h3 {
        margin: 0; font-size: 15px; flex: 1;
    }
    #${MODAL_ID} .c2c-dc-body {
        padding: 12px 18px; overflow-y: auto; flex: 1;
        font-size: 12.5px; line-height: 1.45;
    }
    #${MODAL_ID} .c2c-dc-body section { margin-bottom: 14px; }
    #${MODAL_ID} .c2c-dc-body section h4 {
        margin: 0 0 6px 0; font-size: 12px;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    #${MODAL_ID} .c2c-dc-body .breaking h4 { color: var(--c2c-red); }
    #${MODAL_ID} .c2c-dc-body .risky    h4 { color: var(--c2c-yellow); }
    #${MODAL_ID} .c2c-dc-body ul { margin: 0; padding-left: 18px; }
    #${MODAL_ID} .c2c-dc-body li { margin: 2px 0; }
    #${MODAL_ID} .c2c-dc-body code {
        background: color-mix(in srgb, var(--c2c-highlightBase) 6%, transparent);
        padding: 1px 4px; border-radius: 3px;
        font-size: 11.5px;
    }
    #${MODAL_ID} pre.fix {
        background: var(--c2c-bg3); color: var(--c2c-green);
        padding: 8px 10px; border-radius: 4px;
        font-size: 11.5px; overflow-x: auto;
        border: 1px solid var(--c2c-border);
        white-space: pre-wrap;
    }
    #${MODAL_ID} .c2c-dc-foot {
        padding: 10px 18px; border-top: 1px solid var(--c2c-border);
        display: flex; gap: 8px; justify-content: flex-end;
    }
    #${MODAL_ID} .c2c-dc-btn {
        background: var(--c2c-border); color: var(--c2c-fg);
        border: 1px solid var(--c2c-surface1); border-radius: 4px;
        padding: 6px 12px; font-size: 12px; cursor: pointer;
    }
    #${MODAL_ID} .c2c-dc-btn:hover { background: var(--c2c-surface1); }
    #${MODAL_ID} .c2c-dc-btn.primary {
        background: var(--c2c-blue); color: var(--c2c-bg); border-color: var(--c2c-blue);
        font-weight: 600;
    }
    #${MODAL_ID} .c2c-dc-btn.primary:hover { background: var(--c2c-lavender); }
    `;
    document.head.appendChild(s);
}

function _closeModal() {
    const m = document.getElementById(MODAL_ID);
    if (m) m.remove();
    _modalOpen = false;
}

function _buildFixCommand(reports) {
    // Collect packages whose installed version doesn't satisfy the new
    // requirement. Suggest a `--force-reinstall` of each. Best-effort.
    const pkgs = [];
    for (const r of reports) {
        if (!r.report) continue;
        for (const e of (r.report.breaking || [])) {
            const tok = e.required && e.required !== "(any)"
                ? `${e.package}${e.required.replace(/\s+/g, "")}`
                : e.package;
            if (!pkgs.includes(tok)) pkgs.push(tok);
        }
    }
    if (!pkgs.length) return null;
    return `${SUGGESTED_PIP_HEAD} ${pkgs.join(" ")}`;
}

function _showModal(reports) {
    if (_modalOpen) return;
    _modalOpen = true;
    _injectStyles();

    const overlay = document.createElement("div");
    overlay.id = MODAL_ID;

    const card = document.createElement("div");
    card.className = "c2c-dc-card";

    const head = document.createElement("div");
    head.className = "c2c-dc-head";
    head.innerHTML = `<span class="icon">⚠️</span>
        <h3>Dependency conflict detected after install</h3>`;
    card.appendChild(head);

    const body = document.createElement("div");
    body.className = "c2c-dc-body";

    const intro = document.createElement("p");
    intro.innerHTML = `ComfyUI-Manager just installed a new pack, but its
        <code>requirements.txt</code> conflicts with the package versions
        currently in this environment. ComfyUI may need a forced reinstall
        before the new nodes work reliably.`;
    body.appendChild(intro);

    for (const r of reports) {
        if (!r.report) continue;
        const breaking = r.report.breaking || [];
        const risky    = r.report.risky || [];
        if (!breaking.length && !risky.length) continue;

        const sec = document.createElement("section");
        const head2 = document.createElement("h4");
        head2.textContent = r.pack;
        head2.style.color = "var(--c2c-blue)";
        sec.appendChild(head2);

        if (breaking.length) {
            const bsec = document.createElement("div");
            bsec.className = "breaking";
            bsec.innerHTML = `<h4>Breaking (${breaking.length})</h4>`;
            const ul = document.createElement("ul");
            for (const e of breaking) {
                const li = document.createElement("li");
                li.innerHTML =
                    `<code>${e.package}</code>: installed ` +
                    `<code>${e.installed || "n/a"}</code> → wants ` +
                    `<code>${e.required}</code>`;
                ul.appendChild(li);
            }
            bsec.appendChild(ul);
            sec.appendChild(bsec);
        }
        if (risky.length) {
            const rsec = document.createElement("div");
            rsec.className = "risky";
            rsec.innerHTML = `<h4>Risky (${risky.length})</h4>`;
            const ul = document.createElement("ul");
            for (const e of risky) {
                const li = document.createElement("li");
                li.innerHTML =
                    `<code>${e.package}</code>: installed ` +
                    `<code>${e.installed || "n/a"}</code>, wants ` +
                    `<code>${e.required}</code> — ${e.reason}`;
                ul.appendChild(li);
            }
            rsec.appendChild(ul);
            sec.appendChild(rsec);
        }
        body.appendChild(sec);
    }

    const fixCmd = _buildFixCommand(reports);
    if (fixCmd) {
        const fixSec = document.createElement("section");
        fixSec.innerHTML = `<h4 style="color:var(--c2c-green);">Suggested fix</h4>
            <p style="margin:0 0 6px 0;">Run this in the same Python env
            that hosts ComfyUI:</p>
            <pre class="fix">${fixCmd}</pre>`;
        body.appendChild(fixSec);
    }

    card.appendChild(body);

    const foot = document.createElement("div");
    foot.className = "c2c-dc-foot";

    if (fixCmd) {
        const copy = document.createElement("button");
        copy.className = "c2c-dc-btn primary";
        copy.textContent = "Copy fix command";
        copy.onclick = async () => {
            try {
                await navigator.clipboard.writeText(fixCmd);
                copy.textContent = "Copied ✓";
                setTimeout(() => { copy.textContent = "Copy fix command"; }, 1500);
            } catch { copy.textContent = "Copy failed"; }
        };
        foot.appendChild(copy);
    }
    const dismiss = document.createElement("button");
    dismiss.className = "c2c-dc-btn";
    dismiss.textContent = "Dismiss";
    dismiss.onclick = _closeModal;
    foot.appendChild(dismiss);

    card.appendChild(foot);
    overlay.appendChild(card);
    overlay.addEventListener("click", (e) => {
        if (e.target === overlay) _closeModal();
    });
    document.body.appendChild(overlay);
}

async function _scanAndMaybeWarn() {
    const now = Date.now();
    if (now - _lastScanAt < 4000) return;
    _lastScanAt = now;
    try {
        if (_baseline === null) {
            // No baseline (interceptor fired before bootstrap). Bail.
            return;
        }
        const r = await fetch("/c2c/depcheck/scan_new", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ baseline: _baseline }),
        });
        const j = await r.json().catch(() => null);
        if (!j || !j.success) return;
        const reports = (j.data && j.data.reports) || [];
        const interesting = reports.filter(
            (x) => x.report && ((x.report.breaking || []).length || (x.report.risky || []).length));
        if (interesting.length) _showModal(interesting);
        // refresh baseline so we don't re-warn on the next install for the same pack
        try {
            const snap = await fetch("/c2c/depcheck/snapshot");
            const sj = await snap.json().catch(() => null);
            if (sj && sj.success) _baseline = sj.data.dirs;
        } catch { /* */ }
    } catch (e) {
        console.warn("[c2c.depcheck] scan failed:", e);
    }
}

function _isInstallUrl(url) {
    if (typeof url !== "string") {
        try { url = url.url || url.toString(); } catch { return false; }
    }
    return INSTALL_ENDPOINTS.some((sfx) => url.includes(sfx));
}

function _installFetchInterceptor() {
    const origFetch = window.fetch.bind(window);
    if (origFetch.__c2c_depcheck) return;
    const wrapped = async (input, init) => {
        const url = (typeof input === "string") ? input : (input && input.url) || "";
        const resp = await origFetch(input, init);
        try {
            if (_isInstallUrl(url) && resp.ok) {
                // Wait for Manager to finish unpacking before scanning.
                setTimeout(() => { _scanAndMaybeWarn(); }, 3500);
            }
        } catch { /* never let our hook break the original fetch */ }
        return resp;
    };
    wrapped.__c2c_depcheck = true;
    window.fetch = wrapped;
}

async function _bootstrapBaseline() {
    try {
        const r = await fetch("/c2c/depcheck/snapshot");
        const j = await r.json().catch(() => null);
        if (j && j.success) _baseline = j.data.dirs;
        else _baseline = [];
    } catch {
        _baseline = [];
    }
}

app.registerExtension({
    name: "C2C.DepConflictModal",
    settings: [
        {
            id: "c2c.depcheck.enabled",
            name: "Warn about pip conflicts after Manager install",
            type: "boolean",
            defaultValue: true,
            tooltip: "When ComfyUI-Manager finishes installing a pack, check its requirements.txt against the live environment and pop a modal listing any breaking or risky version conflicts.",
            onChange: () => { /* toggle is read at install time */ },
        },
    ],
    async setup() {
        try {
            const enabled = app.ui?.settings?.getSettingValue?.(
                "c2c.depcheck.enabled", true) ?? true;
            if (!enabled) return;
            await _bootstrapBaseline();
            _installFetchInterceptor();
        } catch (e) {
            console.warn("[c2c.depcheck] setup failed:", e);
        }
    },
});
