/**
 * c2c_scheduled_queue.js — Scheduled Queue + Webhooks panel (rebuilt L9.39).
 * Schedules: cron jobs that queue a workflow on this server (the current graph, or an API JSON file).
 * Webhooks: an inbound trigger URL per stored workflow, and outbound URLs notified when runs finish.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { attachWindowChrome } from "./_c2c_window.js";
import { getRuntime } from "./_c2c_runtime.js";

let _panel  = null;
let _isOpen = false;
let _refreshTimer = null;
let _previewTimer = null;

const _esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function _call(path, body) {
  const r = await api.fetchApi(path, body === undefined ? undefined
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  let data = {};
  try { data = await r.json(); } catch { data = { error: `HTTP ${r.status}` }; }
  if (!r.ok && !data.error) data.error = `HTTP ${r.status}`;
  return data;
}

/** The API prompt of the graph on screen (subgraphs flattened by the front-end). */
async function _currentPrompt() {
  const p = await app.graphToPrompt();
  return p?.output ?? null;
}

function _when(ts) {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  const today = new Date().toDateString() === d.toDateString();
  return today ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : d.toLocaleString([], { weekday: "short", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

const _INPUT = "width:100%;box-sizing:border-box;background:var(--c2c-bg3);border:1px solid var(--c2c-border);border-radius:5px;padding:5px 8px;color:inherit;font-size:12px;";
const _LABEL = "font-size:10px;color:var(--c2c-gray400);margin-bottom:3px;";
const _BTN = (tone) => `padding:6px 12px;background:color-mix(in srgb, var(--c2c-${tone}) 15%, transparent);border:1px solid color-mix(in srgb, var(--c2c-${tone}) 30%, transparent);border-radius:5px;cursor:pointer;color:var(--c2c-fg);font-size:12px;font-weight:600;white-space:nowrap;`;
const _SMALL = "font-size:10px;padding:2px 7px;background:color-mix(in srgb, var(--c2c-surface0) 60%, transparent);border:1px solid var(--c2c-border);border-radius:4px;cursor:pointer;color:var(--c2c-gray300);";
const _ROW = "display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid color-mix(in srgb, var(--c2c-border) 40%, transparent);";
const _SECTION = "padding:8px 16px 4px;font-size:10px;font-weight:700;letter-spacing:.06em;color:var(--c2c-gray400);text-transform:uppercase;";

function _sourceFields(prefix) {
  return `
    <div style="flex:1;min-width:120px;">
      <div style="${_LABEL}">Workflow</div>
      <select id="${prefix}-src" style="${_INPUT}background:var(--c2c-surface0);">
        <option value="current">Current workflow</option>
        <option value="file">API JSON file…</option>
      </select>
    </div>
    <div id="${prefix}-path-wrap" style="flex:2;min-width:160px;" hidden>
      <div style="${_LABEL}">API workflow file (Workflow › Export (API))</div>
      <input id="${prefix}-path" type="text" placeholder="D:\\workflows\\render_api.json" style="${_INPUT}">
    </div>`;
}

function _wireSource(root, prefix) {
  const sel = root.querySelector(`#${prefix}-src`);
  const wrap = root.querySelector(`#${prefix}-path-wrap`);
  sel.addEventListener("change", () => { wrap.hidden = sel.value !== "file"; });
}

/** {workflow} or {workflow_json}, or throws a readable Error. */
async function _readSource(root, prefix) {
  if (root.querySelector(`#${prefix}-src`).value === "file") {
    const path = root.querySelector(`#${prefix}-path`).value.trim();
    if (!path) throw new Error("Enter the path of an API-format workflow file.");
    return { workflow: path };
  }
  const prompt = await _currentPrompt();
  if (!prompt || !Object.keys(prompt).length) throw new Error("The current workflow has no nodes to run.");
  return { workflow_json: prompt };
}

function _buildPanel() {
  if (_panel) return _panel;

  const overlay = document.createElement("div");
  overlay.id = "c2c-scheduled-queue-overlay";
  overlay.style.cssText = `position:fixed;inset:0;background:rgba(0,0,0,.65);z-index:var(--c2c-z-modal);display:flex;align-items:center;justify-content:center;`;
  overlay.addEventListener("click", (e) => { if (e.target === overlay) _close(); });

  const panel = document.createElement("div");
  panel.id = "c2c-scheduled-queue";
  panel.style.cssText = `background:var(--c2c-bg2);border:1px solid var(--c2c-border);border-radius:10px;width:720px;max-width:calc(100vw - 32px);max-height:84vh;display:flex;flex-direction:column;overflow:hidden;box-shadow:0 12px 40px rgba(0,0,0,.6);color:var(--c2c-fg);font-family:var(--c2c-font,system-ui,sans-serif);font-size:13px;`;
  panel.innerHTML = `
    <style>#c2c-scheduled-queue [hidden]{display:none!important}</style>
    <div style="padding:14px 16px;border-bottom:1px solid color-mix(in srgb, var(--c2c-border) 70%, transparent);display:flex;align-items:center;gap:8px;">
      <span style="font-weight:600;font-size:14px;">⏰ Scheduled Queue</span>
      <span id="c2c-sq-server" style="font-size:10px;color:var(--c2c-gray500);"></span>
      <div style="margin-left:auto;display:flex;gap:4px;">
        <button data-tab="schedules" style="${_SMALL}">Schedules</button>
        <button data-tab="webhooks" style="${_SMALL}">Webhooks</button>
      </div>
      <button id="c2c-sq-close" style="background:none;border:none;font-size:18px;cursor:pointer;color:var(--c2c-gray300);">×</button>
    </div>
    <div id="c2c-sq-tab-schedules" style="display:flex;flex-direction:column;min-height:0;flex:1;">
      <div style="padding:12px 16px;border-bottom:1px solid color-mix(in srgb, var(--c2c-border) 50%, transparent);">
        <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:flex-end;">
          <div style="flex:1;min-width:110px;">
            <div style="${_LABEL}">Label</div>
            <input id="c2c-sq-label" type="text" placeholder="Nightly render" style="${_INPUT}">
          </div>
          <div style="flex:1;min-width:130px;">
            <div style="${_LABEL}">Cron (min hour day month weekday)</div>
            <input id="c2c-sq-cron" type="text" placeholder="0 6 * * 1-5" style="${_INPUT}font-family:monospace;">
          </div>
          ${_sourceFields("c2c-sq")}
        </div>
        <div style="display:flex;gap:10px;align-items:center;margin-top:8px;flex-wrap:wrap;">
          <label style="font-size:11px;color:var(--c2c-gray300);display:flex;gap:5px;align-items:center;" title="ComfyUI skips a prompt it has already run. New seeds make every scheduled run a new render.">
            <input id="c2c-sq-reseed" type="checkbox" checked> New seed each run
          </label>
          <span style="font-size:10px;color:var(--c2c-gray600);">Presets:
            <span style="cursor:pointer;color:var(--c2c-blue);" data-preset="0 * * * *">hourly</span> ·
            <span style="cursor:pointer;color:var(--c2c-blue);" data-preset="*/30 * * * *">every 30 min</span> ·
            <span style="cursor:pointer;color:var(--c2c-blue);" data-preset="0 6 * * *">daily 06:00</span> ·
            <span style="cursor:pointer;color:var(--c2c-blue);" data-preset="0 2 * * 1-5">weekdays 02:00</span>
          </span>
          <button id="c2c-sq-add" style="${_BTN("ok")}margin-left:auto;">+ Add job</button>
        </div>
        <div id="c2c-sq-preview" style="font-size:11px;color:var(--c2c-gray400);margin-top:6px;min-height:14px;"></div>
        <div id="c2c-sq-error" style="font-size:11px;color:var(--c2c-dangerSoft);margin-top:2px;" hidden></div>
      </div>
      <div id="c2c-sq-jobs" style="overflow-y:auto;flex:1;padding:4px 16px;min-height:60px;"></div>
      <div id="c2c-sq-history" style="padding:6px 16px;font-size:10px;color:var(--c2c-gray400);border-top:1px solid color-mix(in srgb, var(--c2c-border) 50%, transparent);max-height:90px;overflow-y:auto;"></div>
    </div>
    <div id="c2c-sq-tab-webhooks" style="display:flex;flex-direction:column;min-height:0;flex:1;overflow-y:auto;" hidden>
      <div style="${_SECTION}">Inbound — a URL that queues a stored workflow</div>
      <div style="padding:4px 16px 10px;">
        <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:flex-end;">
          <div style="flex:1;min-width:110px;">
            <div style="${_LABEL}">Name (used in the URL)</div>
            <input id="c2c-wh-id" type="text" placeholder="render-shot" style="${_INPUT}">
          </div>
          ${_sourceFields("c2c-wh")}
          <div style="flex:1;min-width:110px;">
            <div style="${_LABEL}">Secret (optional, HMAC-SHA256)</div>
            <input id="c2c-wh-secret" type="password" autocomplete="new-password" style="${_INPUT}">
          </div>
          <button id="c2c-wh-add" style="${_BTN("ok")}">+ Register</button>
        </div>
        <div id="c2c-wh-error" style="font-size:11px;color:var(--c2c-dangerSoft);margin-top:4px;" hidden></div>
        <div id="c2c-wh-in" style="margin-top:6px;"></div>
      </div>
      <div style="${_SECTION}border-top:1px solid color-mix(in srgb, var(--c2c-border) 50%, transparent);">Outbound — notify a URL when runs finish</div>
      <div style="padding:4px 16px 10px;">
        <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:flex-end;">
          <div style="flex:3;min-width:200px;">
            <div style="${_LABEL}">URL</div>
            <input id="c2c-wo-url" type="text" placeholder="https://hooks.example.com/comfy" style="${_INPUT}">
          </div>
          <div style="flex:1;min-width:110px;">
            <div style="${_LABEL}">Secret (optional)</div>
            <input id="c2c-wo-secret" type="password" autocomplete="new-password" style="${_INPUT}">
          </div>
          <button id="c2c-wo-add" style="${_BTN("ok")}">+ Add</button>
        </div>
        <div id="c2c-wo-events" style="display:flex;gap:10px;flex-wrap:wrap;margin-top:6px;font-size:11px;color:var(--c2c-gray300);">
          ${[["execution_success", "finished", true], ["execution_error", "failed", true], ["execution_interrupted", "interrupted", true],
             ["execution_start", "started", false], ["executed", "each output node", false], ["progress", "every step (chatty)", false]]
            .map(([v, l, on]) => `<label style="display:flex;gap:4px;align-items:center;"><input type="checkbox" value="${v}" ${on ? "checked" : ""}>${l}</label>`).join("")}
        </div>
        <div id="c2c-wo-error" style="font-size:11px;color:var(--c2c-dangerSoft);margin-top:4px;" hidden></div>
        <div id="c2c-wo-list" style="margin-top:6px;"></div>
      </div>
      <div style="${_SECTION}border-top:1px solid color-mix(in srgb, var(--c2c-border) 50%, transparent);">Recent calls</div>
      <div id="c2c-wh-log" style="padding:2px 16px 12px;font-size:10px;color:var(--c2c-gray400);"></div>
    </div>
  `;
  overlay.appendChild(panel);
  document.body.appendChild(overlay);
  _panel = overlay;

  panel.querySelector("#c2c-sq-close").addEventListener("click", _close);
  panel.querySelector("#c2c-sq-add").addEventListener("click", _addJobUI);
  panel.querySelector("#c2c-wh-add").addEventListener("click", _addInboundUI);
  panel.querySelector("#c2c-wo-add").addEventListener("click", _addOutboundUI);
  panel.querySelectorAll("[data-preset]").forEach((sp) => sp.addEventListener("click", () => {
    panel.querySelector("#c2c-sq-cron").value = sp.dataset.preset;
    _previewCron();
  }));
  panel.querySelector("#c2c-sq-cron").addEventListener("input", () => {
    clearTimeout(_previewTimer);
    _previewTimer = setTimeout(_previewCron, 250);
  });
  panel.querySelectorAll("[data-tab]").forEach((b) => b.addEventListener("click", () => _showTab(b.dataset.tab)));
  _wireSource(panel, "c2c-sq");
  _wireSource(panel, "c2c-wh");

  const _header  = panel.querySelector("#c2c-sq-close").parentElement;
  attachWindowChrome(panel, { storageKey: "scheduled_queue", overlay, header: _header,
    titleEl: _header.querySelector("span"), minW: 520, minH: 360 });
  _showTab("schedules");
  return overlay;
}

function _showTab(tab) {
  const p = _panel;
  if (!p) return;
  p.querySelector("#c2c-sq-tab-schedules").hidden = tab !== "schedules";
  p.querySelector("#c2c-sq-tab-webhooks").hidden = tab !== "webhooks";
  p.querySelectorAll("[data-tab]").forEach((b) => {
    b.style.borderColor = b.dataset.tab === tab ? "var(--c2c-blue)" : "var(--c2c-border)";
    b.style.color = b.dataset.tab === tab ? "var(--c2c-fg)" : "var(--c2c-gray300)";
  });
  _refreshUI();
}

function _showError(id, msg) {
  const el = _panel?.querySelector(id);
  if (!el) return;
  el.textContent = msg || "";
  el.hidden = !msg;
}

async function _previewCron() {
  const el = _panel?.querySelector("#c2c-sq-preview");
  const cron = _panel?.querySelector("#c2c-sq-cron")?.value?.trim();
  if (!el) return;
  if (!cron) { el.textContent = ""; return; }
  const r = await _call(`/c2c/schedule/preview?cron=${encodeURIComponent(cron)}`);
  el.style.color = r.ok ? "var(--c2c-gray400)" : "var(--c2c-dangerSoft)";
  el.textContent = r.ok ? `Next: ${(r.next || []).map(_when).join(" · ") || "never"}` : r.error;
}

async function _addJobUI() {
  const p = _panel;
  _showError("#c2c-sq-error", "");
  const cron = p.querySelector("#c2c-sq-cron").value.trim();
  if (!cron) { _showError("#c2c-sq-error", "Enter a cron expression, or pick a preset."); return; }
  try {
    const src = await _readSource(p, "c2c-sq");
    const r = await _call("/c2c/schedule/add", { label: p.querySelector("#c2c-sq-label").value.trim(), cron,
      enabled: true, reseed: p.querySelector("#c2c-sq-reseed").checked, ...src });
    if (r.error) throw new Error(r.error);
    p.querySelector("#c2c-sq-label").value = "";
    await _refreshUI();
  } catch (err) {
    _showError("#c2c-sq-error", err.message || String(err));
  }
}

async function _addInboundUI() {
  const p = _panel;
  _showError("#c2c-wh-error", "");
  try {
    const src = await _readSource(p, "c2c-wh");
    const r = await _call("/c2c/webhook/register", { id: p.querySelector("#c2c-wh-id").value.trim(),
      secret: p.querySelector("#c2c-wh-secret").value, reseed: true, ...src });
    if (r.error) throw new Error(r.error);
    if (r.secret_storage === "memory") _showError("#c2c-wh-error", "No secrets backend: the secret is kept in memory and is lost on restart.");
    p.querySelector("#c2c-wh-id").value = "";
    p.querySelector("#c2c-wh-secret").value = "";
    await _refreshUI();
  } catch (err) {
    _showError("#c2c-wh-error", err.message || String(err));
  }
}

async function _addOutboundUI() {
  const p = _panel;
  _showError("#c2c-wo-error", "");
  const events = [...p.querySelectorAll("#c2c-wo-events input:checked")].map((i) => i.value);
  const r = await _call("/c2c/webhook/outbound/add", { url: p.querySelector("#c2c-wo-url").value.trim(),
    secret: p.querySelector("#c2c-wo-secret").value, events });
  if (r.error) { _showError("#c2c-wo-error", r.error); return; }
  p.querySelector("#c2c-wo-url").value = "";
  p.querySelector("#c2c-wo-secret").value = "";
  await _refreshUI();
}

async function _refreshSchedules() {
  const [listData, histData] = await Promise.all([_call("/c2c/schedule/list"), _call("/c2c/schedule/history")]);
  const p = _panel;
  const srv = p.querySelector("#c2c-sq-server");
  if (srv) srv.textContent = listData.server ? `queues on ${listData.server}` : "";
  const jobsEl = p.querySelector("#c2c-sq-jobs");
  const jobs = listData.jobs || [];
  if (!jobs.length) {
    jobsEl.innerHTML = `<div style="color:var(--c2c-gray600);font-size:11px;padding:10px 0;">No scheduled jobs yet. Pick a cron and "Current workflow", then Add job.</div>`;
  } else {
    jobsEl.innerHTML = jobs.map((j) => {
      const bad = j.last_status === "error";
      return `
        <div style="${_ROW}">
          <span style="width:10px;height:10px;border-radius:50%;background:${j.enabled ? "var(--c2c-okBright)" : "var(--c2c-gray600)"};flex-shrink:0;" title="${j.enabled ? "active" : "paused"}"></span>
          <div style="flex:1;min-width:0;">
            <div style="font-size:12px;font-weight:600;">${_esc(j.label)} <span style="font-weight:400;color:var(--c2c-gray500);font-size:10px;">${_esc(j.source)}${j.nodes ? ` · ${j.nodes} nodes` : ""}${j.reseed ? " · new seeds" : ""}</span></div>
            <div style="font-size:10px;color:var(--c2c-gray400);"><span style="font-family:monospace;">${_esc(j.cron)}</span> · next ${j.enabled ? _when(j.next_run) : "paused"}</div>
            <div style="font-size:10px;color:var(--c2c-gray600);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="${_esc(j.last_error || "")}">Last: ${_when(j.last_run)} <span style="color:${bad ? "var(--c2c-dangerSoft)" : "var(--c2c-okSoft2)"}">${_esc(j.last_status || "")}${bad && j.last_error ? " — " + _esc(j.last_error) : ""}</span></div>
          </div>
          <button data-toggle="${_esc(j.id)}" style="${_SMALL}">${j.enabled ? "Pause" : "Resume"}</button>
          <button data-run="${_esc(j.id)}" style="${_SMALL}color:var(--c2c-okSoft2);">▶ Now</button>
          <button data-remove="${_esc(j.id)}" style="${_SMALL}color:var(--c2c-dangerSoft);" title="Remove">✕</button>
        </div>`;
    }).join("");
    jobsEl.querySelectorAll("[data-toggle]").forEach((b) => b.addEventListener("click", async () => { await _call("/c2c/schedule/toggle", { id: b.dataset.toggle }); _refreshUI(); }));
    jobsEl.querySelectorAll("[data-run]").forEach((b) => b.addEventListener("click", async () => {
      const r = await _call("/c2c/schedule/run_now", { id: b.dataset.run });
      _showError("#c2c-sq-error", r.error || "");
      _refreshUI();
    }));
    jobsEl.querySelectorAll("[data-remove]").forEach((b) => b.addEventListener("click", async () => { await _call("/c2c/schedule/remove", { id: b.dataset.remove }); _refreshUI(); }));
  }
  const hist = (histData.history || []).slice(-6).reverse();
  p.querySelector("#c2c-sq-history").innerHTML = hist.length
    ? `<div style="font-weight:600;margin-bottom:2px;">Recent runs</div>` + hist.map((h) =>
      `<div>${_when(h.started_at)} · ${_esc(h.label)} · ${_esc(h.reason)} · <span style="color:${h.status === "error" ? "var(--c2c-dangerSoft)" : "var(--c2c-okSoft2)"}">${_esc(h.status)}</span>${h.error ? " — " + _esc(h.error) : ""}</div>`).join("")
    : "No runs yet.";
}

async function _copy(text, btn) {
  try { await navigator.clipboard.writeText(text); btn.textContent = "Copied"; }
  catch { btn.textContent = "Copy failed"; }
  setTimeout(() => { btn.textContent = "Copy"; }, 1200);
}

async function _refreshWebhooks() {
  const [inData, outData, logData] = await Promise.all([_call("/c2c/webhook/list"), _call("/c2c/webhook/outbound/list"), _call("/c2c/webhook/log")]);
  const p = _panel;
  const inEl = p.querySelector("#c2c-wh-in");
  const ins = inData.webhooks || [];
  inEl.innerHTML = ins.length ? ins.map((w) => `
    <div style="${_ROW}">
      <div style="flex:1;min-width:0;">
        <div style="font-size:12px;font-weight:600;">${_esc(w.label)} <span style="font-weight:400;color:var(--c2c-gray500);font-size:10px;">${_esc(w.source)}${w.has_secret ? " · signed" : " · no secret"}</span></div>
        <div style="font-size:10px;color:var(--c2c-gray400);font-family:monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="${_esc(w.url)}">POST ${_esc(w.url)}</div>
      </div>
      <button data-copy="${_esc(w.url)}" style="${_SMALL}">Copy</button>
      ${w.has_secret ? "" : `<button data-fire="${_esc(w.id)}" style="${_SMALL}color:var(--c2c-okSoft2);">▶ Trigger</button>`}
      <button data-unreg="${_esc(w.id)}" style="${_SMALL}color:var(--c2c-dangerSoft);" title="Remove">✕</button>
    </div>`).join("")
    : `<div style="color:var(--c2c-gray600);font-size:11px;padding:6px 0;">No inbound webhooks. Register one to get a URL that queues the workflow.</div>`;
  inEl.querySelectorAll("[data-copy]").forEach((b) => b.addEventListener("click", () => _copy(b.dataset.copy, b)));
  inEl.querySelectorAll("[data-fire]").forEach((b) => b.addEventListener("click", async () => {
    const r = await _call(`/c2c/webhook/trigger/${encodeURIComponent(b.dataset.fire)}`, {});
    _showError("#c2c-wh-error", r.error || "");
    _refreshUI();
  }));
  inEl.querySelectorAll("[data-unreg]").forEach((b) => b.addEventListener("click", async () => { await _call("/c2c/webhook/unregister", { id: b.dataset.unreg }); _refreshUI(); }));

  const outEl = p.querySelector("#c2c-wo-list");
  const outs = outData.outbound || [];
  outEl.innerHTML = outs.length ? outs.map((o) => `
    <div style="${_ROW}">
      <div style="flex:1;min-width:0;">
        <div style="font-size:11px;font-family:monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="${_esc(o.url)}">${_esc(o.url)}</div>
        <div style="font-size:10px;color:var(--c2c-gray500);">${(o.events || []).map(_esc).join(", ")}${o.has_secret ? " · signed" : ""}</div>
      </div>
      <button data-test="${_esc(o.id)}" style="${_SMALL}">Send test</button>
      <button data-oremove="${_esc(o.id)}" style="${_SMALL}color:var(--c2c-dangerSoft);" title="Remove">✕</button>
    </div>`).join("")
    : `<div style="color:var(--c2c-gray600);font-size:11px;padding:6px 0;">No outbound URLs.</div>`;
  outEl.querySelectorAll("[data-test]").forEach((b) => b.addEventListener("click", async () => {
    b.textContent = "Sending…";
    const r = await _call("/c2c/webhook/outbound/test", { id: b.dataset.test });
    b.textContent = r.status && /^2\d\d$/.test(r.status) ? `OK ${r.status}` : `Failed ${r.status || ""}`;
    _showError("#c2c-wo-error", r.msg || "");
    setTimeout(() => { b.textContent = "Send test"; _refreshUI(); }, 1500);
  }));
  outEl.querySelectorAll("[data-oremove]").forEach((b) => b.addEventListener("click", async () => { await _call("/c2c/webhook/outbound/remove", { id: b.dataset.oremove }); _refreshUI(); }));

  const logs = (logData.log || []).slice(-8).reverse();
  p.querySelector("#c2c-wh-log").innerHTML = logs.length ? logs.map((l) =>
    `<div>${_when(l.ts)} · ${_esc(l.direction)} · ${_esc(l.id)} · ${_esc(l.event)} · <span style="color:${/^2\d\d$|queued/.test(l.status) ? "var(--c2c-okSoft2)" : "var(--c2c-dangerSoft)"}">${_esc(l.status)}</span>${l.msg ? " — " + _esc(l.msg) : ""}</div>`).join("")
    : "No calls yet.";
}

async function _refreshUI() {
  if (!_panel || !_isOpen) return;
  try {
    if (!_panel.querySelector("#c2c-sq-tab-schedules").hidden) await _refreshSchedules();
    else await _refreshWebhooks();
  } catch (err) {
    console.warn("[C2C.ScheduledQueue] refresh failed:", err);
  }
}

function _open() {
  _buildPanel();
  _panel.style.display = "flex";
  _isOpen = true;
  _refreshUI();
  // Not ambient: the user opened this to watch runs, and ambient timers pause while a prompt runs and in lite mode.
  _refreshTimer = getRuntime().every("schedule.ui", 5000, () => _refreshUI(), { ambient: false });
}

function _close() {
  if (_panel) _panel.style.display = "none";
  _isOpen = false;
  if (_refreshTimer) { _refreshTimer.cancel(); _refreshTimer = null; }
}

function _buildSlot() {
  const btn = document.createElement("button");
  btn.id = "c2c-scheduled-queue-btn";
  btn.className = "c2c-omnibar-slot-pill";
  btn.textContent = "⏰ Schedule";
  btn.title = "Scheduled Queue and Webhooks";
  btn.style.cssText = `font-size:11px;padding:2px 7px;cursor:pointer;background:color-mix(in srgb, var(--c2c-surface1) 90%, transparent);color:var(--c2c-fg);border:1px solid color-mix(in srgb, var(--c2c-border) 24%, transparent);border-radius:10px;`;
  btn.addEventListener("click", () => _isOpen ? _close() : _open());
  return btn;
}

app.registerExtension({
  name: "C2C.ScheduledQueue",
  // The command palette entry works where the OmniBar is not loaded (lite mode strips it).
  commands: [{ id: "C2C.scheduledQueue.open", label: "C2C: Scheduled Queue & Webhooks", function: () => _open() }],
  async setup() {
    // OmniBar is optional (lite mode has none): retry briefly, then stop - the panel stays reachable through
    // window.C2CScheduledQueue.open().
    let tries = 0;
    const tryRegister = () => {
      if (window.C2COmniBar) {
        window.C2COmniBar.register({ section: "tools", id: "c2c-scheduled-queue", order: 45, element: _buildSlot() });
      } else if (++tries < 50) {
        setTimeout(tryRegister, 200);
      }
    };
    tryRegister();
    window.C2CScheduledQueue = { open: _open, close: _close };
  },
});
