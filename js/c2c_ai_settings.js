/**
 * c2c_ai_settings.js — Settings & first-run wizard for the C2C AI spine.
 *
 * Adds:
 *   1. A sidebar tab "C2C AI" with three sub-views:
 *        - Backends     (add/remove/test, choose model, enable toggle)
 *        - Routing      (per-feature policy matrix)
 *        - Privacy/$$   (daily cap, redaction preview, keys manager)
 *   2. A first-run wizard modal that runs once if ai_config.json is empty.
 *      Detects local servers, offers to import keys from
 *      `All API Keys Of Comfy.txt`, shows the keychain explanation
 *      (verbatim) before doing so, then asks whether to delete the source
 *      .txt file.
 */
import { app } from "../../scripts/app.js";
import { buildPanel } from "./_c2c_window.js";

const TAB_ID = "c2c.ai";
const RIGHT_DOCK_ID = "c2c.ai.right-dock";
const SETTING_FIRSTRUN = "c2c.ai.firstRunCompleted";

// All colors source from CSS custom properties emitted by _c2c_theme.js
// (--c2c-bg, --c2c-mauve, etc.) — modal, sidebar, wizard all repaint when
// setVariant() flips mocha/latte/oled. No hardcoded palette.

const VERBATIM_KEYCHAIN_NOTICE =
`Importing your API keys

C2C found "All API Keys Of Comfy.txt" in your project folder. We can save these keys into your operating system's secure credential store (Windows Credential Manager / macOS Keychain / Linux Secret Service) so they're encrypted at rest and only readable by your user account.

What happens next:
  1. We read the file once.
  2. Each key (ANTHROPIC_API_KEY, OPENAI_API_KEY, QWEN_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY, COHERE_API_KEY, AZURE_OPENAI_API_KEY, HF_TOKEN) is stored as a separate entry in the credential store under the service name "c2c-comfy".
  3. The original .txt file is NOT automatically deleted — you'll be asked next.`;

// --- low-level API helpers ---------------------------------------------------
async function api(method, url, body) {
    const opts = { method, headers: { "Content-Type": "application/json" }};
    if (body !== undefined) opts.body = JSON.stringify(body);
    const r = await fetch(url, opts);
    const j = await r.json().catch(() => ({}));
    if (!j.success) throw new Error(j.message || `${method} ${url} failed`);
    return j.data;
}
const apiGet  = (u)    => api("GET",  u);
const apiPost = (u, b) => api("POST", u, b);

// HTML-attribute-safe escape (used when interpolating user/server values
// into ``data-*`` attributes inside template-literal HTML — prevents the
// model id "claude-3-5-sonnet" or a quoted display name from breaking out
// of the attribute and injecting markup).
function escAttr(s) {
    return String(s ?? "")
        .replace(/&/g, "&amp;")
        .replace(/"/g, "&quot;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

// =============================================================== modal helpers
function modal({ title, body, buttons }) {
    return new Promise((resolve) => {
        const back = document.createElement("div");
        back.style.cssText =
            `position:fixed;inset:0;z-index:var(--c2c-z-modal);background:color-mix(in srgb, var(--c2c-shadowBase) 55%, transparent);
             display:flex;align-items:center;justify-content:center;`;
        const card = document.createElement("div");
        card.style.cssText =
            `background:var(--c2c-bg);color:var(--c2c-fg);border:1px solid var(--c2c-border);
             border-radius:8px;padding:18px;min-width:480px;max-width:720px;
             max-height:80vh;overflow:auto;
             font-family:ui-sans-serif,system-ui,sans-serif;font-size:13px;
             box-shadow:0 12px 40px color-mix(in srgb, var(--c2c-shadowBase) 60%, transparent);`;
        card.innerHTML =
            `<h2 style="margin:0 0 10px;color:var(--c2c-mauve);font-size:15px;
                        text-transform:uppercase;letter-spacing:0.5px;">${title}</h2>`;
        const bodyDiv = document.createElement("div");
        if (typeof body === "string") bodyDiv.innerHTML = body;
        else bodyDiv.appendChild(body);
        card.appendChild(bodyDiv);

        const btnRow = document.createElement("div");
        btnRow.style.cssText = "display:flex;gap:8px;justify-content:flex-end;margin-top:14px;";
        (buttons || []).forEach(b => {
            const btn = document.createElement("button");
            btn.textContent = b.label;
            btn.style.cssText =
                `background:${b.primary ? "var(--c2c-mauve)" : "var(--c2c-bg2)"};
                 color:${b.primary ? "var(--c2c-bg)" : "var(--c2c-fg)"};
                 border:1px solid ${b.primary ? "var(--c2c-mauve)" : "var(--c2c-border)"};
                 border-radius:4px;padding:6px 14px;cursor:pointer;font-size:12px;`;
            btn.onclick = () => { back.remove(); resolve(b.value); };
            btnRow.appendChild(btn);
        });
        card.appendChild(btnRow);
        back.appendChild(card);
        document.body.appendChild(back);
    });
}

function preBlock(text) {
    const pre = document.createElement("pre");
    pre.textContent = text;
    pre.style.cssText =
        `background:var(--c2c-bg2);border:1px solid var(--c2c-border);border-radius:4px;
         padding:10px;font-size:11px;white-space:pre-wrap;color:var(--c2c-sub);
         max-height:280px;overflow:auto;line-height:1.45;font-family:ui-monospace,monospace;`;
    return pre;
}

// ================================================================== Wizard
async function runFirstRunWizard() {
    // Step 1 — Welcome
    const start = await modal({
        title: "Welcome to C2C AI (v2.0-dev)",
        body:
`<p>C2C v2.0 adds AI-powered helpers across many features. Every AI call goes
through one router — pick <b>Cloud</b>, <b>Local</b>, or let it auto-choose
per feature. You can change this any time in <i>Settings → C2C ▸ AI Backends</i>.</p>
<p style="color:var(--c2c-sub)">This wizard will:
 1. Detect any local AI servers (Ollama, LM Studio, llama.cpp).
 2. Offer to import any API keys you have in <code>All API Keys Of Comfy.txt</code>.
 3. Save a starting configuration. Nothing is sent anywhere until you actually use an AI feature.</p>`,
        buttons: [
            { label: "Skip — I'll set up later", value: "skip" },
            { label: "Continue", value: "go", primary: true },
        ],
    });
    if (start !== "go") {
        await apiPost("/c2c/ai/config", { version:1, backends:[], first_run_completed:true });
        app.ui?.settings?.setSettingValue(SETTING_FIRSTRUN, true);
        return;
    }

    // Step 2 — Detect local
    let detect;
    try {
        detect = await apiGet("/c2c/ai/local/detect");
    } catch (e) {
        detect = { servers: [] };
    }
    const localList = detect.servers || [];
    const localBody = document.createElement("div");
    if (localList.length === 0) {
        localBody.innerHTML =
            `<p>No local AI servers detected on the usual ports
             (Ollama:11434, LM Studio:1234, llama.cpp:8080, vLLM:8000).</p>
             <p style="color:var(--c2c-sub)">If you have Ollama installed, start it and re-run this
             wizard from <i>Settings → C2C ▸ AI Backends → Re-run wizard</i>.</p>`;
    } else {
        localBody.innerHTML =
            `<p>Local AI servers detected:</p>
             <ul>${localList.map(s =>
                `<li><b>${s.name}</b> @ ${s.base_url}
                 ${s.models?.length ? `<div style="color:var(--c2c-sub);font-size:11px">
                 models: ${s.models.slice(0,3).join(", ")}${s.models.length>3 ? "…" : ""}</div>` : ""}</li>`).join("")}</ul>`;
    }
    const continueLocal = await modal({
        title: "Step 1 of 3 — Local servers",
        body: localBody,
        buttons: [{ label: "Continue", value: true, primary: true }],
    });
    void continueLocal;

    // Step 3 — Import keys from txt
    const txtPathCandidate = "d:\\PROJECT\\Custom_Nodes\\All API Keys Of Comfy.txt";
    const importChoice = await modal({
        title: "Step 2 of 3 — API Keys",
        body: (() => {
            const div = document.createElement("div");
            div.innerHTML =
                `<p>Cloud providers need API keys. We can read them from your existing
                 file and store them in your OS credential store (encrypted at rest).</p>
                 <p style="color:var(--c2c-sub);font-size:11px">Suggested file:
                 <code>${txtPathCandidate}</code></p>`;
            div.appendChild(preBlock(VERBATIM_KEYCHAIN_NOTICE));
            return div;
        })(),
        buttons: [
            { label: "Skip", value: "skip" },
            { label: "Enter keys manually later", value: "skip" },
            { label: "Import from txt", value: "import", primary: true },
        ],
    });

    let imported = [];
    if (importChoice === "import") {
        try {
            const res = await apiPost("/c2c/ai/keys/import_txt", { path: txtPathCandidate });
            imported = res.imported || [];
        } catch (exc) {
            await modal({
                title: "Import failed",
                body: `<p style="color:var(--c2c-red)">${String(exc.message || exc)}</p>
                       <p style="color:var(--c2c-sub)">You can paste keys manually under
                       <i>Settings → C2C ▸ AI Backends → Keys</i>.</p>`,
                buttons: [{ label: "Continue", value: true, primary: true }],
            });
        }
        if (imported.length > 0) {
            const del = await modal({
                title: "Imported successfully",
                body: `<p>Stored in keychain:</p><ul>${imported.map(k => `<li><code>${k}</code></li>`).join("")}</ul>
                       <p>Delete the original <code>All API Keys Of Comfy.txt</code> file now? (recommended — the keys are now safer in the credential store)</p>`,
                buttons: [
                    { label: "Keep .txt file", value: "keep" },
                    { label: "Delete .txt file", value: "delete", primary: true },
                ],
            });
            if (del === "delete") {
                // We don't have a backend route for file delete (out of scope for v1).
                // Surface a clear instruction instead so the user can rm it themselves.
                await modal({
                    title: "Delete the file",
                    body: `<p>To finish, please delete the file manually:</p>
                           <p><code>${txtPathCandidate}</code></p>
                           <p style="color:var(--c2c-sub)">C2C deliberately won't auto-delete user files.</p>`,
                    buttons: [{ label: "Done", value: true, primary: true }],
                });
            }
        }
    }

    // Step 4 — Build initial config
    const haveKeys = await apiGet("/c2c/ai/keys/list").catch(() => ({ keys: [] }));
    const startCfg = { version:1, backends:[], first_run_completed:true };
    if (haveKeys.keys?.includes("ANTHROPIC_API_KEY")) {
        startCfg.backends.push({ kind:"anthropic", id:"cloud.anthropic",
            model:"claude-3-5-sonnet-latest", enabled:true });
    }
    if (haveKeys.keys?.includes("OPENAI_API_KEY")) {
        startCfg.backends.push({ kind:"openai", id:"cloud.openai",
            model:"gpt-4o-mini", enabled:true });
    }
    if (haveKeys.keys?.includes("QWEN_API_KEY")) {
        startCfg.backends.push({ kind:"qwen", id:"cloud.qwen",
            model:"qwen-max-latest", enabled:false });
    }
    if (haveKeys.keys?.includes("OPENROUTER_API_KEY")) {
        startCfg.backends.push({ kind:"openrouter", id:"cloud.openrouter",
            model:"anthropic/claude-3.5-sonnet", enabled:false });
    }
    if (haveKeys.keys?.includes("GEMINI_API_KEY")) {
        startCfg.backends.push({ kind:"gemini", id:"cloud.gemini",
            model:"gemini-1.5-flash-latest", enabled:true });
    }
    if (haveKeys.keys?.includes("COHERE_API_KEY")) {
        startCfg.backends.push({ kind:"cohere", id:"cloud.cohere",
            model:"command-r-08-2024", enabled:false });
    }
    // Azure OpenAI is NOT auto-enabled at first-run even when the key is
    // present because it additionally requires endpoint + deployment name
    // that we can't auto-detect. The user adds it manually in the editor.
    (detect.servers || []).slice(0, 2).forEach(srv => {
        startCfg.backends.push({
            kind:"local", id:srv.id, display_name:srv.name,
            base_url:srv.base_url,
            model: srv.models?.[0] || "auto",
            max_context: 32768, enabled:true,
        });
    });

    await apiPost("/c2c/ai/config", startCfg);
    await modal({
        title: "Step 3 of 3 — Ready",
        body: `<p>Initial config saved:</p>
               <ul>${startCfg.backends.map(b => `<li>${b.kind} — <code>${b.id}</code>${b.enabled ? "" : " (disabled)"}</li>`).join("") || "<li>(none yet)</li>"}</ul>
               <p style="color:var(--c2c-sub)">Tweak any time under <i>Settings → C2C ▸ AI Backends</i>.</p>`,
        buttons: [{ label: "Finish", value: true, primary: true }],
    });
    app.ui?.settings?.setSettingValue(SETTING_FIRSTRUN, true);
    window.__C2C_AI_HUD__?.refresh?.();
}

// ============================================================== sidebar tab
function buildSettingsView(root) {
    root.innerHTML = "";
    root.style.cssText =
        `padding:12px;color:var(--c2c-fg);background:var(--c2c-bg);
         font-family:ui-sans-serif,system-ui,sans-serif;font-size:12px;height:100%;
         overflow:auto;`;

    const header = document.createElement("div");
    header.innerHTML =
        `<h3 style="margin:0 0 4px;color:var(--c2c-mauve);text-transform:uppercase;
                    letter-spacing:0.6px;font-size:13px;">C2C AI Backends</h3>
         <div style="color:var(--c2c-sub);font-size:11px;margin-bottom:10px;">
            One spine. Cloud and Local share the same router.
         </div>`;
    root.appendChild(header);

    const sections = document.createElement("div");
    root.appendChild(sections);

    async function refresh() {
        sections.innerHTML = "<div style='color:var(--c2c-sub)'>loading…</div>";
        let st, cfg;
        try {
            [st, cfg] = await Promise.all([apiGet("/c2c/ai/status"), apiGet("/c2c/ai/config")]);
        } catch (exc) {
            sections.innerHTML =
                `<div style="color:var(--c2c-red)">Could not load status: ${exc.message}</div>`;
            return;
        }

        sections.innerHTML = "";

        // ----- Backends section
        const sec1 = document.createElement("section");
        sec1.innerHTML = `<h4 style="color:var(--c2c-blue);margin:8px 0 6px;font-size:12px;">Backends</h4>`;
        const tbl = document.createElement("table");
        tbl.style.cssText = "width:100%;border-collapse:collapse;font-size:11px;";
        tbl.innerHTML =
            `<thead><tr>
               <th style="text-align:left;padding:4px;border-bottom:1px solid var(--c2c-border)">id</th>
               <th style="text-align:left;padding:4px;border-bottom:1px solid var(--c2c-border)">model</th>
               <th style="padding:4px;border-bottom:1px solid var(--c2c-border)">health</th>
               <th style="padding:4px;border-bottom:1px solid var(--c2c-border)">enabled</th>
               <th style="padding:4px;border-bottom:1px solid var(--c2c-border)"></th>
             </tr></thead>`;
        const tbody = document.createElement("tbody");
        (st.backends || []).forEach(b => {
            const tr = document.createElement("tr");
            tr.innerHTML =
                `<td style="padding:4px;border-bottom:1px solid var(--c2c-border)">
                   <span style="color:${b.tier==="cloud"?"var(--c2c-blue)":"var(--c2c-green)"}">${b.tier==="cloud"?"☁":"💻"}</span>
                   ${b.id}<div style="color:var(--c2c-sub);font-size:10px">${b.display_name}</div></td>
                 <td style="padding:4px;border-bottom:1px solid var(--c2c-border)" class="c2c-model-cell" data-id="${b.id}" data-current="${escAttr(b.model || "")}">
                   <span style="color:var(--c2c-sub)">${b.model}</span></td>
                 <td style="padding:4px;border-bottom:1px solid var(--c2c-border);text-align:center;
                     color:${b.health?.ok ? "var(--c2c-green)" : "var(--c2c-red)"}">
                   ${b.health?.ok ? "ok" : "down"}
                   <div style="color:var(--c2c-sub);font-size:10px">${b.health?.last_rtt_ms ?? "?"}ms</div></td>
                 <td style="padding:4px;border-bottom:1px solid var(--c2c-border);text-align:center">
                   <input type="checkbox" data-id="${b.id}" class="c2c-en" ${b.enabled?"checked":""}></td>
                 <td style="padding:4px;border-bottom:1px solid var(--c2c-border)">
                   <button data-id="${b.id}" class="c2c-test" style="background:var(--c2c-bg2);color:var(--c2c-fg);
                     border:1px solid var(--c2c-border);border-radius:3px;padding:2px 8px;cursor:pointer;font-size:10px">test</button></td>`;
            tbody.appendChild(tr);
        });
        tbl.appendChild(tbody);
        sec1.appendChild(tbl);

        const btnRow = document.createElement("div");
        btnRow.style.cssText = "margin-top:8px;display:flex;gap:6px;flex-wrap:wrap;";
        ["Refresh probes", "Re-run wizard"].forEach(label => {
            const b = document.createElement("button");
            b.textContent = label;
            b.style.cssText =
                `background:var(--c2c-bg2);color:var(--c2c-fg);border:1px solid var(--c2c-border);
                 border-radius:4px;padding:4px 10px;cursor:pointer;font-size:11px;`;
            b.onclick = async () => {
                if (label === "Refresh probes") { await apiPost("/c2c/ai/probe"); refresh(); }
                else { await runFirstRunWizard(); refresh(); }
            };
            btnRow.appendChild(b);
        });
        // Show "Pop out right" only when this view is NOT already hosted in
        // the right dock (otherwise the button is redundant and the click
        // would simply close the dock via the toggle).
        if (!root.closest?.(`#${RIGHT_DOCK_ID}`)) {
            const popBtn = document.createElement("button");
            popBtn.textContent = "Pop out → right dock";
            popBtn.title = "Open the AI panel as a floating window on the right (Ctrl+Alt+A)";
            popBtn.style.cssText =
                `background:var(--c2c-bg2);color:var(--c2c-fg);border:1px solid var(--c2c-border);
                 border-radius:4px;padding:4px 10px;cursor:pointer;font-size:11px;margin-left:auto;`;
            popBtn.onclick = () => {
                try { openRightDock(); }
                catch (exc) { console.error("[c2c.ai] openRightDock failed:", exc); }
            };
            btnRow.appendChild(popBtn);
        }
        sec1.appendChild(btnRow);
        sections.appendChild(sec1);

        tbody.querySelectorAll("input.c2c-en").forEach(cb => {
            cb.addEventListener("change", async (e) => {
                const id = e.target.dataset.id;
                const newCfg = await apiGet("/c2c/ai/config");
                const entry = (newCfg.backends || []).find(x => x.id === id);
                if (entry) { entry.enabled = e.target.checked; await apiPost("/c2c/ai/config", newCfg); refresh(); }
            });
        });
        tbody.querySelectorAll("button.c2c-test").forEach(b => {
            b.addEventListener("click", async () => {
                b.textContent = "…";
                try {
                    const r = await apiPost("/c2c/ai/backends/test", { id: b.dataset.id });
                    b.textContent = r.ok ? "ok" : "fail";
                    setTimeout(() => { b.textContent = "test"; refresh(); }, 1500);
                } catch (exc) {
                    b.textContent = "err"; setTimeout(() => b.textContent = "test", 1500);
                }
            });
        });

        // --- model-picker hydration (P0.7-2b) -----------------------------
        // For each backend row, fetch the candidate-model list lazily (one
        // request per backend, in parallel) and replace the static label
        // with a <select>. On change, GET /c2c/ai/config, patch entry.model
        // for the matching backend, POST it back, then refresh.
        tbody.querySelectorAll("td.c2c-model-cell").forEach(async (cell) => {
            const id = cell.dataset.id;
            const current = cell.dataset.current || "";
            let res;
            try {
                res = await apiGet(`/c2c/ai/backends/${encodeURIComponent(id)}/models`);
            } catch (_exc) {
                return; // leave static label in place; backend may be unreachable
            }
            const models = Array.isArray(res?.models) ? res.models : [];
            if (models.length <= 1) return; // nothing to pick from — keep label
            const sel = document.createElement("select");
            sel.dataset.id = id;
            sel.className = "c2c-model-select";
            sel.style.cssText =
                `background:var(--c2c-bg2);color:var(--c2c-fg);border:1px solid var(--c2c-border);
                 border-radius:3px;padding:2px 4px;font-size:11px;max-width:220px;`;
            // ensure current model is always present so the active selection
            // never disappears even if the backend dropped it from /models.
            const seen = new Set();
            const ordered = [];
            if (current) { seen.add(current); ordered.push(current); }
            for (const m of models) { if (!seen.has(m)) { seen.add(m); ordered.push(m); } }
            for (const m of ordered) {
                const opt = document.createElement("option");
                opt.value = m; opt.textContent = m;
                if (m === current) opt.selected = true;
                sel.appendChild(opt);
            }
            sel.addEventListener("change", async (e) => {
                const next = e.target.value;
                if (!next || next === current) return;
                e.target.disabled = true;
                try {
                    const cfgNow = await apiGet("/c2c/ai/config");
                    const entry = (cfgNow.backends || []).find(x => x.id === id);
                    if (!entry) throw new Error(`backend ${id} missing from config`);
                    entry.model = next;
                    await apiPost("/c2c/ai/config", cfgNow);
                    refresh();
                } catch (exc) {
                    e.target.disabled = false;
                    e.target.value = current;
                    console.error("[c2c.ai] model swap failed:", exc);
                    alert(`Could not switch model: ${exc.message || exc}`);
                }
            });
            cell.innerHTML = "";
            cell.appendChild(sel);
        });

        // ----- Routing section
        const sec2 = document.createElement("section");
        sec2.innerHTML = `<h4 style="color:var(--c2c-blue);margin:14px 0 6px;font-size:12px;">Per-feature routing</h4>`;
        const policyTbl = document.createElement("table");
        policyTbl.style.cssText = "width:100%;border-collapse:collapse;font-size:11px;";
        (st.policies || []).forEach(p => {
            const tr = document.createElement("tr");
            const sel = `<select data-feature="${p.feature}" class="c2c-policy" style="background:var(--c2c-bg2);color:var(--c2c-fg);border:1px solid var(--c2c-border);border-radius:3px;font-size:11px;padding:2px 4px;">
              ${["", "auto","prefer_local","prefer_cloud","cloud_only","local_only"].map(v =>
                `<option value="${v}" ${(p.override||"")===v ? "selected" : ""}>${v||"(use default)"}</option>`).join("")}
            </select>`;
            tr.innerHTML =
                `<td style="padding:4px;border-bottom:1px solid var(--c2c-border)">${p.feature}</td>
                 <td style="padding:4px;border-bottom:1px solid var(--c2c-border);color:var(--c2c-sub);font-size:10px">default: ${p.default}</td>
                 <td style="padding:4px;border-bottom:1px solid var(--c2c-border)">${sel}</td>
                 <td style="padding:4px;border-bottom:1px solid var(--c2c-border);color:var(--c2c-mauve)">→ ${p.effective}</td>`;
            policyTbl.appendChild(tr);
        });
        sec2.appendChild(policyTbl);
        sections.appendChild(sec2);
        policyTbl.querySelectorAll("select.c2c-policy").forEach(sel => {
            sel.addEventListener("change", async () => {
                await apiPost("/c2c/ai/policy", {
                    feature: sel.dataset.feature,
                    policy: sel.value || null,
                });
                refresh();
            });
        });

        // ----- Cost section
        const sec3 = document.createElement("section");
        const cost = st.cost || {};
        sec3.innerHTML =
            `<h4 style="color:var(--c2c-blue);margin:14px 0 6px;font-size:12px;">Daily budget</h4>
             <div>$${(cost.today_cost_usd ?? 0).toFixed(4)} / cap $${(cost.cap_usd ?? 0).toFixed(2)}
                  (${cost.today_calls ?? 0} calls today)</div>
             <div style="margin-top:8px">
               <label style="color:var(--c2c-sub);font-size:11px">Daily cap (USD):</label>
               <input id="c2c-cap" type="number" step="0.05" min="0" value="${cost.cap_usd ?? 1.0}"
                      style="background:var(--c2c-bg2);color:var(--c2c-fg);border:1px solid var(--c2c-border);
                             border-radius:3px;padding:2px 6px;width:80px;font-size:11px;margin-left:6px"/>
               <button id="c2c-cap-save" style="background:var(--c2c-bg2);color:var(--c2c-fg);border:1px solid var(--c2c-border);
                       border-radius:3px;padding:2px 10px;cursor:pointer;font-size:11px;margin-left:6px">save</button>
             </div>`;
        sections.appendChild(sec3);
        sec3.querySelector("#c2c-cap-save").onclick = async () => {
            const usd = parseFloat(sec3.querySelector("#c2c-cap").value);
            if (!isFinite(usd) || usd < 0) return;
            await apiPost("/c2c/ai/cost/cap", { usd });
            refresh();
        };

        // ----- Local GGUF (text_encoders/) section — B1
        // Pulls the live list of GGUF files from ComfyUI's `text_encoders`
        // folder via /c2c/ai/text_encoders/list and lets the user add one
        // as a `text_encoder_local` backend without editing ai_config.json
        // by hand.
        const sec4 = document.createElement("section");
        sec4.innerHTML =
            `<h4 style="color:var(--c2c-blue);margin:14px 0 6px;font-size:12px;">
                Local GGUF (text_encoders/)
             </h4>
             <div id="c2c-te-status" style="color:var(--c2c-sub);font-size:11px;
                  margin-bottom:6px;">scanning…</div>
             <div id="c2c-te-controls" style="display:flex;gap:6px;flex-wrap:wrap;
                  align-items:center;"></div>
             <div id="c2c-te-roots" style="color:var(--c2c-sub);font-size:10px;
                  margin-top:6px;"></div>`;
        sections.appendChild(sec4);

        (async () => {
            let te;
            try {
                te = await apiGet("/c2c/ai/text_encoders/list");
            } catch (exc) {
                sec4.querySelector("#c2c-te-status").innerHTML =
                    `<span style="color:var(--c2c-red)">scan failed: ${exc.message}</span>`;
                return;
            }
            const statusEl = sec4.querySelector("#c2c-te-status");
            const ctrlEl   = sec4.querySelector("#c2c-te-controls");
            const rootsEl  = sec4.querySelector("#c2c-te-roots");
            const items = te.items || [];
            const haveLib = !!te.llamacpp_available;
            const counts = te.counts || {};
            const chatItems = items.filter(it => it.chat_capable);

            // Status line: lib availability + breakdown.
            statusEl.innerHTML =
                `${haveLib
                    ? `<span style="color:var(--c2c-green)">✓ llama-cpp-python installed</span>`
                    : `<span style="color:var(--c2c-red)">✗ llama-cpp-python not installed</span> ` +
                      `<code style="color:var(--c2c-sub)">pip install llama-cpp-python</code>`}
                 · <span>${items.length} file${items.length===1?"":"s"} total</span>
                 · <span style="color:var(--c2c-green)">${counts.chat_capable ?? 0} GGUF</span>
                 · <span style="color:var(--c2c-sub)">${counts.encoder ?? 0} encoder</span>
                 ${(counts.other ?? 0) > 0
                    ? `· <span style="color:var(--c2c-sub)">${counts.other} other</span>`
                    : ""}`;

            // Roots line.
            rootsEl.innerHTML = `Scanned folders: ${
                (te.roots || []).map(r => `<code>${r}</code>`).join(" · ") || "(none)"
            }`;

            // The "model_file" the active local.llamacpp backend points at,
            // if any — pre-select it in the combo.
            const existing = (cfg.backends || []).find(
                e => e.kind === "text_encoder_local",
            );

            if (items.length === 0) {
                ctrlEl.innerHTML =
                    `<span style="color:var(--c2c-sub);font-size:11px">
                        Drop a *.gguf model into the folder above and click Rescan.
                     </span>`;
            } else {
                const sel = document.createElement("select");
                sel.style.cssText =
                    `background:var(--c2c-bg2);color:var(--c2c-fg);
                     border:1px solid var(--c2c-border);border-radius:3px;
                     font-size:11px;padding:2px 4px;max-width:340px;`;
                items.forEach(it => {
                    const o = document.createElement("option");
                    o.value = it.name;
                    const mb = it.size_bytes
                        ? ` (${(it.size_bytes / (1024 * 1024)).toFixed(0)} MB)`
                        : "";
                    const badge =
                        it.kind === "gguf"     ? "[GGUF]"    :
                        it.kind === "encoder"  ? "[encoder]" : "[other]";
                    o.textContent = `${badge} ${it.name}${mb}`;
                    if (!it.chat_capable) {
                        o.disabled = true;
                        o.title = it.kind === "encoder"
                            ? "Diffusion text-encoder (T5/CLIP). Used by image " +
                              "pipelines — not loadable as an in-process chat LLM."
                            : "Not a recognised model file.";
                    }
                    if (existing && existing.model_file === it.name) {
                        o.selected = true;
                    }
                    sel.appendChild(o);
                });
                // Ensure the initial selection lands on a chat-capable option
                // if nothing was pre-selected and any exist.
                if (!existing && chatItems.length) {
                    sel.value = chatItems[0].name;
                }
                ctrlEl.appendChild(sel);

                const apply = document.createElement("button");
                apply.textContent = existing ? "Update" : "Add as backend";
                apply.style.cssText =
                    `background:var(--c2c-bg2);color:var(--c2c-fg);
                     border:1px solid var(--c2c-border);border-radius:3px;
                     padding:2px 10px;cursor:pointer;font-size:11px;`;
                const disabledReason = !haveLib
                    ? "llama-cpp-python must be installed first"
                    : chatItems.length === 0
                        ? "No GGUF files found — only GGUF can be loaded in-process"
                        : "";
                apply.disabled = !!disabledReason;
                apply.title = disabledReason;
                apply.onclick = async () => {
                    // Guard: only allow chat-capable models through.
                    const picked = items.find(it => it.name === sel.value);
                    if (!picked || !picked.chat_capable) {
                        apply.textContent = "GGUF only";
                        setTimeout(() => {
                            apply.textContent = existing ? "Update" : "Add as backend";
                        }, 1500);
                        return;
                    }
                    const newCfg = await apiGet("/c2c/ai/config");
                    newCfg.backends = newCfg.backends || [];
                    let entry = newCfg.backends.find(
                        e => e.kind === "text_encoder_local",
                    );
                    if (entry) {
                        entry.model_file = sel.value;
                        entry.enabled = true;
                    } else {
                        newCfg.backends.push({
                            kind: "text_encoder_local",
                            id: "local.llamacpp",
                            display_name: "Local GGUF (llama.cpp)",
                            model_file: sel.value,
                            n_ctx: 8192,
                            n_gpu_layers: -1,
                            enabled: true,
                        });
                    }
                    await apiPost("/c2c/ai/config", newCfg);
                    refresh();
                };
                ctrlEl.appendChild(apply);
            }

            const rescan = document.createElement("button");
            rescan.textContent = "Rescan";
            rescan.style.cssText =
                `background:var(--c2c-bg2);color:var(--c2c-fg);
                 border:1px solid var(--c2c-border);border-radius:3px;
                 padding:2px 10px;cursor:pointer;font-size:11px;`;
            rescan.onclick = () => refresh();
            ctrlEl.appendChild(rescan);
        })();
    }
    refresh();
}

// ============================================================ right-side dock
// ComfyUI's extensionManager.registerSidebarTab() docks every tab on the
// LEFT only (there is no `side` or `position` option). To deliver the
// "AI panel on the right" form factor that the P0.7 brief requires, we
// build a floating panel via the shared _c2c_window.buildPanel() and
// position it at the right viewport edge on first open. The same
// buildSettingsView() renders into the floating body so the dock + the
// left sidebar tab always show identical content.
function openRightDock() {
    const existing = document.getElementById(RIGHT_DOCK_ID);
    if (existing) {
        // toggle: second activation closes the dock
        existing.remove();
        return null;
    }
    const width = 420;
    const height = Math.max(360, Math.min(window.innerHeight - 80, 900));
    const panel = buildPanel({
        id: RIGHT_DOCK_ID,
        title: "C2C AI",
        shortcut: "Ctrl+Alt+A",
        width,
        height,
        storageKey: "ai-right-dock",
    });
    // On first open (no persisted position yet), park against right edge.
    // buildPanel's cascade() leaves explicit left/top; only override when
    // the storageKey has no saved geometry.
    try {
        const persisted = localStorage.getItem("c2c.win.ai-right-dock");
        if (!persisted) {
            panel.el.style.left = `${Math.max(0, window.innerWidth - width - 16)}px`;
            panel.el.style.top  = `60px`;
        }
    } catch (_exc) { /* ignore quota / privacy errors */ }
    panel.body.style.padding = "0";
    panel.body.style.overflow = "auto";
    buildSettingsView(panel.body);
    if (panel.setSubtitle) panel.setSubtitle("right dock");
    return panel;
}

// Global Ctrl+Alt+A keyboard toggle. Registered once at module load —
// guard with a flag so re-imports don't stack listeners.
if (!window.__C2C_AI_DOCK_HOTKEY__) {
    window.__C2C_AI_DOCK_HOTKEY__ = true;
    window.addEventListener("keydown", (ev) => {
        // Ignore when typing into a text field — the shortcut must not
        // steal characters from a prompt textarea.
        const t = ev.target;
        const tag = (t && t.tagName) || "";
        const editable =
            tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" ||
            (t && t.isContentEditable);
        if (editable) return;
        if (ev.ctrlKey && ev.altKey && !ev.shiftKey && !ev.metaKey &&
            (ev.key === "a" || ev.key === "A")) {
            ev.preventDefault();
            try { openRightDock(); }
            catch (exc) { console.error("[c2c.ai] openRightDock failed:", exc); }
        }
    }, true);
}

// ============================================================ registration
app.registerExtension({
    name: "c2c.ai.settings",
    settings: [
        { id: SETTING_FIRSTRUN, name: "C2C ▸ AI ▸ First-run completed",
          type: "hidden", default: false },
    ],
    async setup() {
        try {
            app.extensionManager?.registerSidebarTab?.({
                id: TAB_ID,
                icon: "pi pi-cog",
                title: "C2C AI",
                tooltip: "C2C AI Backends — configure cloud + local providers",
                type: "custom",
                render: (el) => buildSettingsView(el),
            });
        } catch (exc) {
            console.warn("[c2c.ai.settings] sidebar tab registration deferred:", exc);
        }

        // First-run wizard — only if backends are empty AND user hasn't dismissed
        try {
            const cfg = await apiGet("/c2c/ai/config");
            const completed = app.ui?.settings?.getSettingValue(SETTING_FIRSTRUN, false);
            if (!completed && !cfg.first_run_completed) {
                // small delay to let UI settle
                setTimeout(() => runFirstRunWizard().catch(console.error), 1500);
            }
        } catch (exc) {
            console.warn("[c2c.ai.settings] firstrun check failed:", exc);
        }
    },
});

// expose for other JS that wants to re-trigger the wizard or pop the dock
window.__C2C_AI_SETTINGS__ = { runFirstRunWizard, openRightDock };
