/**
 * c2c_watch_folder.js — Watch Folder UI (P2.1)
 * Monitor folders for new files and auto-queue workflows.
 * Open via OmniBar tools slot.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { attachWindowChrome } from "./_c2c_window.js";

let _panel  = null;
let _isOpen = false;
let _refreshTimer = null;

/* ─── API ─── */

async function _getStatus()          { return (await api.fetchApi("/c2c/watchfolder/status")).json(); }
async function _getEvents()          { return (await api.fetchApi("/c2c/watchfolder/events")).json(); }
async function _addWatcher(body)     { return (await api.fetchApi("/c2c/watchfolder/add", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body) })).json(); }
async function _removeWatcher(path)  { return (await api.fetchApi("/c2c/watchfolder/remove", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({ path }) })).json(); }

/* ─── panel ─── */

function _buildPanel() {
  if (_panel) return _panel;

  const overlay = document.createElement("div");
  overlay.id = "c2c-watch-folder-overlay";
  overlay.style.cssText = `
    position:fixed;inset:0;background:rgba(0,0,0,.65);z-index:var(--c2c-z-modal);
    display:flex;align-items:center;justify-content:center;
  `;
  overlay.addEventListener("click", (e) => { if (e.target === overlay) _close(); });

  const panel = document.createElement("div");
  panel.id = "c2c-watch-folder";
  panel.style.cssText = `
    background:var(--c2c-surface,var(--c2c-neutral950));border:1px solid var(--c2c-border,rgba(255,255,255,.15));
    border-radius:10px;width:640px;max-height:80vh;display:flex;flex-direction:column;
    overflow:hidden;box-shadow:0 12px 40px rgba(0,0,0,.6);
    color:var(--c2c-fg,var(--c2c-gray100));font-family:var(--c2c-font,system-ui,sans-serif);font-size:13px;
  `;
  panel.innerHTML = `
    <div style="padding:14px 16px;border-bottom:1px solid rgba(255,255,255,.1);display:flex;align-items:center;gap:8px;">
      <span style="font-weight:600;font-size:14px;">👁 Watch Folder</span>
      <button id="c2c-wf-close" style="background:none;border:none;font-size:18px;cursor:pointer;color:var(--c2c-gray300);margin-left:auto;">×</button>
    </div>
    <div style="padding:12px 16px;border-bottom:1px solid rgba(255,255,255,.06);">
      <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:flex-end;">
        <div style="flex:2;min-width:140px;">
          <div style="font-size:10px;color:var(--c2c-gray400);margin-bottom:3px;">Folder path</div>
          <input id="c2c-wf-path" type="text" placeholder="C:\\path\\to\\folder"
            style="width:100%;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.12);
                   border-radius:5px;padding:5px 8px;color:inherit;font-size:12px;">
        </div>
        <div style="flex:1;min-width:110px;">
          <div style="font-size:10px;color:var(--c2c-gray400);margin-bottom:3px;">Action</div>
          <select id="c2c-wf-action"
            style="width:100%;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.12);
                   border-radius:5px;padding:5px 6px;color:inherit;font-size:12px;">
            <option value="notify">Notify only</option>
            <option value="load_image">Load image</option>
            <option value="load_workflow">Run workflow</option>
          </select>
        </div>
        <div id="c2c-wf-workflow-wrap" style="flex:2;min-width:140px;display:none;">
          <div style="font-size:10px;color:var(--c2c-gray400);margin-bottom:3px;">Workflow JSON path</div>
          <input id="c2c-wf-workflow" type="text" placeholder="path/to/workflow.json"
            style="width:100%;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.12);
                   border-radius:5px;padding:5px 8px;color:inherit;font-size:12px;">
        </div>
        <button id="c2c-wf-add"
          style="padding:6px 14px;background:rgba(0,200,100,.15);border:1px solid rgba(0,200,100,.3);
                 border-radius:5px;cursor:pointer;color:var(--c2c-okSoft2);font-size:12px;font-weight:600;white-space:nowrap;">
          + Add Watcher
        </button>
      </div>
    </div>
    <div id="c2c-wf-watchers" style="padding:8px 16px;border-bottom:1px solid rgba(255,255,255,.06);min-height:40px;"></div>
    <div style="padding:6px 16px;font-size:10px;color:var(--c2c-gray400);font-weight:600;border-bottom:1px solid rgba(255,255,255,.04);">
      RECENT EVENTS
    </div>
    <div id="c2c-wf-events" style="overflow-y:auto;flex:1;padding:6px 16px;font-size:11px;"></div>
  `;
  overlay.appendChild(panel);
  document.body.appendChild(overlay);
  _panel = overlay;

  panel.querySelector("#c2c-wf-close").addEventListener("click", _close);
  panel.querySelector("#c2c-wf-add").addEventListener("click", _addWatcherUI);
  panel.querySelector("#c2c-wf-action").addEventListener("change", (e) => {
    const wrap = panel.querySelector("#c2c-wf-workflow-wrap");
    wrap.style.display = e.target.value === "load_workflow" ? "" : "none";
  });

  const _header  = panel.firstElementChild;
  const _titleEl = _header?.querySelector("span");
  attachWindowChrome(panel, {
    storageKey: "watch_folder",
    overlay,
    header: _header,
    titleEl: _titleEl,
    minW: 480, minH: 320,
  });

  return overlay;
}

async function _addWatcherUI() {
  const path   = document.querySelector("#c2c-wf-path")?.value?.trim();
  const action = document.querySelector("#c2c-wf-action")?.value || "notify";
  const wfPath = document.querySelector("#c2c-wf-workflow")?.value?.trim() || null;
  if (!path) return;
  try {
    await _addWatcher({ path, action, workflow_path: wfPath });
    document.querySelector("#c2c-wf-path").value = "";
    await _refreshUI();
  } catch (err) {
    console.error("[C2C.WatchFolder]", err);
  }
}

async function _refreshUI() {
  const [statusData, eventsData] = await Promise.all([_getStatus(), _getEvents()]);

  const watchersEl = document.getElementById("c2c-wf-watchers");
  if (watchersEl) {
    if (!statusData.watchers?.length) {
      watchersEl.innerHTML = `<div style="color:var(--c2c-gray600);font-size:11px;padding:6px 0;">No active watchers.</div>`;
    } else {
      watchersEl.innerHTML = statusData.watchers.map(w => `
        <div style="display:flex;align-items:center;gap:8px;padding:4px 0;border-bottom:1px solid rgba(255,255,255,.04);">
          <span style="flex:1;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${w.path}">${w.path}</span>
          <span style="font-size:10px;color:var(--c2c-gray400);">${w.action}</span>
          <span style="width:8px;height:8px;border-radius:50%;background:${w.running ? "var(--c2c-okBright)" : "var(--c2c-gray400)"};display:inline-block;"></span>
          <button data-remove="${w.path}"
            style="font-size:10px;padding:2px 7px;background:rgba(255,80,80,.1);border:1px solid rgba(255,80,80,.2);
                   border-radius:4px;cursor:pointer;color:var(--c2c-dangerSoft);">Remove</button>
        </div>
      `).join("");
      watchersEl.querySelectorAll("[data-remove]").forEach(btn => {
        btn.addEventListener("click", async () => {
          await _removeWatcher(btn.dataset.remove);
          await _refreshUI();
        });
      });
    }
  }

  const eventsEl = document.getElementById("c2c-wf-events");
  if (eventsEl) {
    const events = (eventsData.events || []).slice(-30).reverse();
    if (!events.length) {
      eventsEl.innerHTML = `<div style="color:var(--c2c-gray600);padding:8px 0;">No events yet.</div>`;
    } else {
      eventsEl.innerHTML = events.map(ev => {
        const d = new Date(ev.timestamp * 1000);
        const t = d.toLocaleTimeString();
        const name = ev.path.split(/[\\/]/).pop();
        return `<div style="padding:3px 0;border-bottom:1px solid rgba(255,255,255,.03);display:flex;gap:8px;">
          <span style="color:var(--c2c-gray600);min-width:60px;">${t}</span>
          <span style="color:var(--c2c-blue);">${ev.type}</span>
          <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--c2c-gray250);" title="${ev.path}">${name}</span>
        </div>`;
      }).join("");
    }
  }
}

function _open() {
  _buildPanel();
  _panel.style.display = "flex";
  _isOpen = true;
  _refreshUI();
  _refreshTimer = setInterval(_refreshUI, 3000);
}

function _close() {
  if (_panel) _panel.style.display = "none";
  _isOpen = false;
  if (_refreshTimer) { clearInterval(_refreshTimer); _refreshTimer = null; }
}

function _buildSlot() {
  const btn = document.createElement("button");
  btn.id        = "c2c-watch-folder-btn";
  btn.className = "c2c-omnibar-slot-pill";
  btn.textContent = "👁 Watch";
  btn.title     = "Watch Folder";
  btn.style.cssText = `
    font-size:11px;padding:2px 7px;cursor:pointer;
    background:var(--c2c-pill-bg,rgba(255,255,255,.07));
    color:var(--c2c-fg,var(--c2c-gray200));border:1px solid var(--c2c-border,rgba(255,255,255,.12));border-radius:10px;
  `;
  btn.addEventListener("click", () => _isOpen ? _close() : _open());
  return btn;
}

app.registerExtension({
  name: "C2C.WatchFolder",
  async setup() {
    const tryRegister = () => {
      if (!window.C2COmniBar) return setTimeout(tryRegister, 200);
      window.C2COmniBar.register({ section:"tools", id:"c2c-watch-folder", order:40, element:_buildSlot() });
    };
    tryRegister();
    window.C2CWatchFolder = { open: _open, close: _close };
  }
});
