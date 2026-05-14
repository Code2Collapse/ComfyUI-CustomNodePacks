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

const TAB_ID = "c2c.ai";
const SETTING_FIRSTRUN = "c2c.ai.firstRunCompleted";

const C = {
    bg:"#1e1e2e", bg2:"#181825", fg:"#cdd6f4", sub:"#a6adc8",
    border:"#313244", red:"#f38ba8", green:"#a6e3a1", yellow:"#f9e2af",
    blue:"#89b4fa", mauve:"#cba6f7", teal:"#94e2d5",
};

const VERBATIM_KEYCHAIN_NOTICE =
`Importing your API keys

C2C found "All API Keys Of Comfy.txt" in your project folder. We can save these keys into your operating system's secure credential store (Windows Credential Manager / macOS Keychain / Linux Secret Service) so they're encrypted at rest and only readable by your user account.

What happens next:
  1. We read the file once.
  2. Each key (ANTHROPIC_API_KEY, OPENAI_API_KEY, QWEN_API_KEY, OPENROUTER_API_KEY, HF_TOKEN) is stored as a separate entry in the credential store under the service name "c2c-comfy".
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

// =============================================================== modal helpers
function modal({ title, body, buttons }) {
    return new Promise((resolve) => {
        const back = document.createElement("div");
        back.style.cssText =
            `position:fixed;inset:0;z-index:20000;background:rgba(0,0,0,0.55);
             display:flex;align-items:center;justify-content:center;`;
        const card = document.createElement("div");
        card.style.cssText =
            `background:${C.bg};color:${C.fg};border:1px solid ${C.border};
             border-radius:8px;padding:18px;min-width:480px;max-width:720px;
             max-height:80vh;overflow:auto;
             font-family:ui-sans-serif,system-ui,sans-serif;font-size:13px;
             box-shadow:0 12px 40px rgba(0,0,0,0.6);`;
        card.innerHTML =
            `<h2 style="margin:0 0 10px;color:${C.mauve};font-size:15px;
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
                `background:${b.primary ? C.mauve : C.bg2};
                 color:${b.primary ? C.bg : C.fg};
                 border:1px solid ${b.primary ? C.mauve : C.border};
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
        `background:${C.bg2};border:1px solid ${C.border};border-radius:4px;
         padding:10px;font-size:11px;white-space:pre-wrap;color:${C.sub};
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
<p style="color:${C.sub}">This wizard will:
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
             <p style="color:${C.sub}">If you have Ollama installed, start it and re-run this
             wizard from <i>Settings → C2C ▸ AI Backends → Re-run wizard</i>.</p>`;
    } else {
        localBody.innerHTML =
            `<p>Local AI servers detected:</p>
             <ul>${localList.map(s =>
                `<li><b>${s.name}</b> @ ${s.base_url}
                 ${s.models?.length ? `<div style="color:${C.sub};font-size:11px">
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
                 <p style="color:${C.sub};font-size:11px">Suggested file:
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
                body: `<p style="color:${C.red}">${String(exc.message || exc)}</p>
                       <p style="color:${C.sub}">You can paste keys manually under
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
                           <p style="color:${C.sub}">C2C deliberately won't auto-delete user files.</p>`,
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
               <p style="color:${C.sub}">Tweak any time under <i>Settings → C2C ▸ AI Backends</i>.</p>`,
        buttons: [{ label: "Finish", value: true, primary: true }],
    });
    app.ui?.settings?.setSettingValue(SETTING_FIRSTRUN, true);
    window.__C2C_AI_HUD__?.refresh?.();
}

// ============================================================== sidebar tab
function buildSettingsView(root) {
    root.innerHTML = "";
    root.style.cssText =
        `padding:12px;color:${C.fg};background:${C.bg};
         font-family:ui-sans-serif,system-ui,sans-serif;font-size:12px;height:100%;
         overflow:auto;`;

    const header = document.createElement("div");
    header.innerHTML =
        `<h3 style="margin:0 0 4px;color:${C.mauve};text-transform:uppercase;
                    letter-spacing:0.6px;font-size:13px;">C2C AI Backends</h3>
         <div style="color:${C.sub};font-size:11px;margin-bottom:10px;">
            One spine. Cloud and Local share the same router.
         </div>`;
    root.appendChild(header);

    const sections = document.createElement("div");
    root.appendChild(sections);

    async function refresh() {
        sections.innerHTML = "<div style='color:" + C.sub + "'>loading…</div>";
        let st, cfg;
        try {
            [st, cfg] = await Promise.all([apiGet("/c2c/ai/status"), apiGet("/c2c/ai/config")]);
        } catch (exc) {
            sections.innerHTML =
                `<div style="color:${C.red}">Could not load status: ${exc.message}</div>`;
            return;
        }

        sections.innerHTML = "";

        // ----- Backends section
        const sec1 = document.createElement("section");
        sec1.innerHTML = `<h4 style="color:${C.blue};margin:8px 0 6px;font-size:12px;">Backends</h4>`;
        const tbl = document.createElement("table");
        tbl.style.cssText = "width:100%;border-collapse:collapse;font-size:11px;";
        tbl.innerHTML =
            `<thead><tr>
               <th style="text-align:left;padding:4px;border-bottom:1px solid ${C.border}">id</th>
               <th style="text-align:left;padding:4px;border-bottom:1px solid ${C.border}">model</th>
               <th style="padding:4px;border-bottom:1px solid ${C.border}">health</th>
               <th style="padding:4px;border-bottom:1px solid ${C.border}">enabled</th>
               <th style="padding:4px;border-bottom:1px solid ${C.border}"></th>
             </tr></thead>`;
        const tbody = document.createElement("tbody");
        (st.backends || []).forEach(b => {
            const tr = document.createElement("tr");
            tr.innerHTML =
                `<td style="padding:4px;border-bottom:1px solid ${C.border}">
                   <span style="color:${b.tier==="cloud"?C.blue:C.green}">${b.tier==="cloud"?"☁":"💻"}</span>
                   ${b.id}<div style="color:${C.sub};font-size:10px">${b.display_name}</div></td>
                 <td style="padding:4px;border-bottom:1px solid ${C.border};color:${C.sub}">${b.model}</td>
                 <td style="padding:4px;border-bottom:1px solid ${C.border};text-align:center;
                     color:${b.health?.ok ? C.green : C.red}">
                   ${b.health?.ok ? "ok" : "down"}
                   <div style="color:${C.sub};font-size:10px">${b.health?.last_rtt_ms ?? "?"}ms</div></td>
                 <td style="padding:4px;border-bottom:1px solid ${C.border};text-align:center">
                   <input type="checkbox" data-id="${b.id}" class="c2c-en" ${b.enabled?"checked":""}></td>
                 <td style="padding:4px;border-bottom:1px solid ${C.border}">
                   <button data-id="${b.id}" class="c2c-test" style="background:${C.bg2};color:${C.fg};
                     border:1px solid ${C.border};border-radius:3px;padding:2px 8px;cursor:pointer;font-size:10px">test</button></td>`;
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
                `background:${C.bg2};color:${C.fg};border:1px solid ${C.border};
                 border-radius:4px;padding:4px 10px;cursor:pointer;font-size:11px;`;
            b.onclick = async () => {
                if (label === "Refresh probes") { await apiPost("/c2c/ai/probe"); refresh(); }
                else { await runFirstRunWizard(); refresh(); }
            };
            btnRow.appendChild(b);
        });
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

        // ----- Routing section
        const sec2 = document.createElement("section");
        sec2.innerHTML = `<h4 style="color:${C.blue};margin:14px 0 6px;font-size:12px;">Per-feature routing</h4>`;
        const policyTbl = document.createElement("table");
        policyTbl.style.cssText = "width:100%;border-collapse:collapse;font-size:11px;";
        (st.policies || []).forEach(p => {
            const tr = document.createElement("tr");
            const sel = `<select data-feature="${p.feature}" class="c2c-policy" style="background:${C.bg2};color:${C.fg};border:1px solid ${C.border};border-radius:3px;font-size:11px;padding:2px 4px;">
              ${["", "auto","prefer_local","prefer_cloud","cloud_only","local_only"].map(v =>
                `<option value="${v}" ${(p.override||"")===v ? "selected" : ""}>${v||"(use default)"}</option>`).join("")}
            </select>`;
            tr.innerHTML =
                `<td style="padding:4px;border-bottom:1px solid ${C.border}">${p.feature}</td>
                 <td style="padding:4px;border-bottom:1px solid ${C.border};color:${C.sub};font-size:10px">default: ${p.default}</td>
                 <td style="padding:4px;border-bottom:1px solid ${C.border}">${sel}</td>
                 <td style="padding:4px;border-bottom:1px solid ${C.border};color:${C.mauve}">→ ${p.effective}</td>`;
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
            `<h4 style="color:${C.blue};margin:14px 0 6px;font-size:12px;">Daily budget</h4>
             <div>$${(cost.today_cost_usd ?? 0).toFixed(4)} / cap $${(cost.cap_usd ?? 0).toFixed(2)}
                  (${cost.today_calls ?? 0} calls today)</div>
             <div style="margin-top:8px">
               <label style="color:${C.sub};font-size:11px">Daily cap (USD):</label>
               <input id="c2c-cap" type="number" step="0.05" min="0" value="${cost.cap_usd ?? 1.0}"
                      style="background:${C.bg2};color:${C.fg};border:1px solid ${C.border};
                             border-radius:3px;padding:2px 6px;width:80px;font-size:11px;margin-left:6px"/>
               <button id="c2c-cap-save" style="background:${C.bg2};color:${C.fg};border:1px solid ${C.border};
                       border-radius:3px;padding:2px 10px;cursor:pointer;font-size:11px;margin-left:6px">save</button>
             </div>`;
        sections.appendChild(sec3);
        sec3.querySelector("#c2c-cap-save").onclick = async () => {
            const usd = parseFloat(sec3.querySelector("#c2c-cap").value);
            if (!isFinite(usd) || usd < 0) return;
            await apiPost("/c2c/ai/cost/cap", { usd });
            refresh();
        };
    }
    refresh();
}

// ============================================================ registration
app.registerExtension({
    name: "c2c.ai.settings",
    settings: [
        { id: SETTING_FIRSTRUN, name: "C2C ▸ AI ▸ First-run completed",
          type: "hidden", defaultValue: false },
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

// expose for other JS that wants to re-trigger the wizard
window.__C2C_AI_SETTINGS__ = { runFirstRunWizard };
