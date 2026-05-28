/**
 * c2c_scheduled_queue.js — Scheduled Queue UI (P2.1)
 * Cron-style scheduling for ComfyUI workflows.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { attachWindowChrome } from "./_c2c_window.js";

let _panel  = null;
let _isOpen = false;
let _refreshTimer = null;

async function _list()     { return (await api.fetchApi("/c2c/schedule/list")).json(); }
async function _history()  { return (await api.fetchApi("/c2c/schedule/history")).json(); }
async function _add(body)  { return (await api.fetchApi("/c2c/schedule/add", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)})).json(); }
async function _remove(id) { return (await api.fetchApi("/c2c/schedule/remove", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id})})).json(); }
async function _toggle(id) { return (await api.fetchApi("/c2c/schedule/toggle", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id})})).json(); }
async function _runNow(id) { return (await api.fetchApi("/c2c/schedule/run_now", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id})})).json(); }

function _buildPanel() {
  if (_panel) return _panel;

  const overlay = document.createElement("div");
  overlay.id = "c2c-scheduled-queue-overlay";
  overlay.style.cssText = `
    position:fixed;inset:0;background:rgba(0,0,0,.65);z-index:var(--c2c-z-modal);
    display:flex;align-items:center;justify-content:center;
  `;
  overlay.addEventListener("click", (e) => { if (e.target === overlay) _close(); });

  const panel = document.createElement("div");
  panel.style.cssText = `
    background:var(--c2c-surface,var(--c2c-neutral950));border:1px solid var(--c2c-border,rgba(255,255,255,.15));
    border-radius:10px;width:640px;max-height:80vh;display:flex;flex-direction:column;
    overflow:hidden;box-shadow:0 12px 40px rgba(0,0,0,.6);
    color:var(--c2c-fg,var(--c2c-gray100));font-family:var(--c2c-font,system-ui,sans-serif);font-size:13px;
  `;
  panel.innerHTML = `
    <div style="padding:14px 16px;border-bottom:1px solid rgba(255,255,255,.1);display:flex;align-items:center;gap:8px;">
      <span style="font-weight:600;font-size:14px;">⏰ Scheduled Queue</span>
      <button id="c2c-sq-close" style="background:none;border:none;font-size:18px;cursor:pointer;color:var(--c2c-gray300);margin-left:auto;">×</button>
    </div>
    <div style="padding:12px 16px;border-bottom:1px solid rgba(255,255,255,.06);">
      <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:flex-end;">
        <div style="flex:1;min-width:100px;">
          <div style="font-size:10px;color:var(--c2c-gray400);margin-bottom:3px;">Label</div>
          <input id="c2c-sq-label" type="text" placeholder="My Job"
            style="width:100%;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.12);
                   border-radius:5px;padding:5px 8px;color:inherit;font-size:12px;">
        </div>
        <div style="flex:1;min-width:130px;">
          <div style="font-size:10px;color:var(--c2c-gray400);margin-bottom:3px;">Cron (min hr dom mon dow)</div>
          <input id="c2c-sq-cron" type="text" placeholder="0 * * * *"
            style="width:100%;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.12);
                   border-radius:5px;padding:5px 8px;color:inherit;font-size:12px;font-family:monospace;">
        </div>
        <div style="flex:2;min-width:140px;">
          <div style="font-size:10px;color:var(--c2c-gray400);margin-bottom:3px;">Workflow JSON path</div>
          <input id="c2c-sq-workflow" type="text" placeholder="path/to/workflow.json"
            style="width:100%;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.12);
                   border-radius:5px;padding:5px 8px;color:inherit;font-size:12px;">
        </div>
        <button id="c2c-sq-add"
          style="padding:6px 14px;background:rgba(0,200,100,.15);border:1px solid rgba(0,200,100,.3);
                 border-radius:5px;cursor:pointer;color:var(--c2c-okSoft2);font-size:12px;font-weight:600;white-space:nowrap;">
          + Add Job
        </button>
      </div>
      <div style="margin-top:6px;font-size:10px;color:var(--c2c-gray600);">
        Cron presets: <span style="cursor:pointer;color:var(--c2c-blue);" data-preset="0 * * * *">every hour</span> ·
        <span style="cursor:pointer;color:var(--c2c-blue);" data-preset="0 6 * * *">daily 6am</span> ·
        <span style="cursor:pointer;color:var(--c2c-blue);" data-preset="*/30 * * * *">every 30min</span>
      </div>
    </div>
    <div id="c2c-sq-jobs" style="overflow-y:auto;flex:1;padding:8px 16px;"></div>
    <div style="padding:6px 16px;font-size:10px;color:var(--c2c-gray400);border-top:1px solid rgba(255,255,255,.06);">
      Run history: <span id="c2c-sq-history-summary"></span>
    </div>
  `;
  overlay.appendChild(panel);
  document.body.appendChild(overlay);
  _panel = overlay;

  panel.querySelector("#c2c-sq-close").addEventListener("click", _close);
  panel.querySelector("#c2c-sq-add").addEventListener("click", _addJobUI);
  panel.querySelectorAll("[data-preset]").forEach(sp => {
    sp.addEventListener("click", () => {
      document.querySelector("#c2c-sq-cron").value = sp.dataset.preset;
    });
  });

  const _header  = panel.firstElementChild;
  const _titleEl = _header?.querySelector("span");
  attachWindowChrome(panel, {
    storageKey: "scheduled_queue",
    overlay,
    header: _header,
    titleEl: _titleEl,
    minW: 480, minH: 320,
  });

  return overlay;
}

async function _addJobUI() {
  const label    = document.querySelector("#c2c-sq-label")?.value?.trim() || "";
  const cron     = document.querySelector("#c2c-sq-cron")?.value?.trim() || "";
  const workflow = document.querySelector("#c2c-sq-workflow")?.value?.trim() || "";
  if (!cron) return;
  try {
    await _add({ label, cron, workflow, enabled: true });
    document.querySelector("#c2c-sq-label").value = "";
    document.querySelector("#c2c-sq-cron").value = "";
    document.querySelector("#c2c-sq-workflow").value = "";
    await _refreshUI();
  } catch (err) {
    console.error("[C2C.ScheduledQueue]", err);
  }
}

async function _refreshUI() {
  const [listData, histData] = await Promise.all([_list(), _history()]);

  const jobsEl = document.getElementById("c2c-sq-jobs");
  if (!jobsEl) return;
  const jobs = listData.jobs || [];
  if (!jobs.length) {
    jobsEl.innerHTML = `<div style="color:var(--c2c-gray600);font-size:11px;padding:8px 0;">No scheduled jobs.</div>`;
  } else {
    jobsEl.innerHTML = jobs.map(j => {
      const lastRun = j.last_run ? new Date(j.last_run * 1000).toLocaleString() : "never";
      const statusCol = j.last_status?.startsWith("error") ? "var(--c2c-dangerSoft)" : "var(--c2c-okSoft2)";
      return `
        <div style="display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid rgba(255,255,255,.04);">
          <span style="width:10px;height:10px;border-radius:50%;background:${j.enabled ? "var(--c2c-okBright)" : "var(--c2c-gray600)"};flex-shrink:0;"></span>
          <div style="flex:1;min-width:0;">
            <div style="font-size:12px;font-weight:600;">${j.label || j.id}</div>
            <div style="font-size:10px;color:var(--c2c-gray400);font-family:monospace;">${j.cron}</div>
            <div style="font-size:10px;color:var(--c2c-gray600);">Last: ${lastRun} <span style="color:${statusCol}">${j.last_status || ""}</span></div>
          </div>
          <button data-toggle="${j.id}"
            style="font-size:10px;padding:2px 7px;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.12);
                   border-radius:4px;cursor:pointer;color:var(--c2c-gray300);">${j.enabled ? "Pause" : "Resume"}</button>
          <button data-run="${j.id}"
            style="font-size:10px;padding:2px 7px;background:rgba(0,200,100,.12);border:1px solid rgba(0,200,100,.25);
                   border-radius:4px;cursor:pointer;color:var(--c2c-okSoft2);">▶ Now</button>
          <button data-remove="${j.id}"
            style="font-size:10px;padding:2px 7px;background:rgba(255,80,80,.1);border:1px solid rgba(255,80,80,.2);
                   border-radius:4px;cursor:pointer;color:var(--c2c-dangerSoft);">✕</button>
        </div>
      `;
    }).join("");
    jobsEl.querySelectorAll("[data-toggle]").forEach(b => b.addEventListener("click", async () => { await _toggle(b.dataset.toggle); await _refreshUI(); }));
    jobsEl.querySelectorAll("[data-run]").forEach(b => b.addEventListener("click", async () => { await _runNow(b.dataset.run); await _refreshUI(); }));
    jobsEl.querySelectorAll("[data-remove]").forEach(b => b.addEventListener("click", async () => { await _remove(b.dataset.remove); await _refreshUI(); }));
  }

  const hist = histData.history || [];
  const summEl = document.getElementById("c2c-sq-history-summary");
  if (summEl) summEl.textContent = hist.length ? `${hist.length} runs` : "none";
}

function _open() {
  _buildPanel();
  _panel.style.display = "flex";
  _isOpen = true;
  _refreshUI();
  _refreshTimer = setInterval(_refreshUI, 5000);
}

function _close() {
  if (_panel) _panel.style.display = "none";
  _isOpen = false;
  if (_refreshTimer) { clearInterval(_refreshTimer); _refreshTimer = null; }
}

function _buildSlot() {
  const btn = document.createElement("button");
  btn.id = "c2c-scheduled-queue-btn";
  btn.className = "c2c-omnibar-slot-pill";
  btn.textContent = "⏰ Schedule";
  btn.title = "Scheduled Queue";
  btn.style.cssText = `
    font-size:11px;padding:2px 7px;cursor:pointer;
    background:var(--c2c-pill-bg,rgba(255,255,255,.07));
    color:var(--c2c-fg,var(--c2c-gray200));border:1px solid var(--c2c-border,rgba(255,255,255,.12));border-radius:10px;
  `;
  btn.addEventListener("click", () => _isOpen ? _close() : _open());
  return btn;
}

app.registerExtension({
  name: "C2C.ScheduledQueue",
  async setup() {
    const tryRegister = () => {
      if (!window.C2COmniBar) return setTimeout(tryRegister, 200);
      window.C2COmniBar.register({ section:"tools", id:"c2c-scheduled-queue", order:45, element:_buildSlot() });
    };
    tryRegister();
    window.C2CScheduledQueue = { open: _open, close: _close };
  }
});
