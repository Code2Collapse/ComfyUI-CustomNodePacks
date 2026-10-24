/**
 * c2c_watch_folder.js — Watch Folder panel (rebuilt L9.39).
 * Watch a folder for new image / video files: log them, copy them into input/c2c_watch and offer a Load node, or
 * run a workflow on each one (the current graph or an API JSON file). Open via the OmniBar tools slot.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { attachWindowChrome } from "./_c2c_window.js";
import { getRuntime } from "./_c2c_runtime.js";

let _panel  = null;
let _isOpen = false;
let _refreshTimer = null;

const _esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
/** The end of a long path ("…\renders\incoming") - the folder name is the part that tells watchers apart. */
const _shortPath = (p, max = 64) => {
  const s = String(p ?? "");
  if (s.length <= max) return s;
  const parts = s.split(/([\\/])/);
  let out = "";
  for (let i = parts.length - 1; i >= 0 && (out.length + parts[i].length) < max - 1; i--) out = parts[i] + out;
  return "…" + (out || s.slice(-(max - 1)));
};
const _INPUT = "width:100%;box-sizing:border-box;background:var(--c2c-bg3);border:1px solid var(--c2c-border);border-radius:5px;padding:5px 8px;color:inherit;font-size:12px;";
const _LABEL = "font-size:10px;color:var(--c2c-gray400);margin-bottom:3px;";
const _SMALL = "font-size:10px;padding:2px 7px;background:color-mix(in srgb, var(--c2c-surface0) 60%, transparent);border:1px solid var(--c2c-border);border-radius:4px;cursor:pointer;color:var(--c2c-gray300);";
const _EVENT_TONE = { new_file: "var(--c2c-blue)", copied: "var(--c2c-okSoft2)", queued: "var(--c2c-okSoft2)", error: "var(--c2c-dangerSoft)", too_many_files: "var(--c2c-warn, #d9a441)" };

async function _call(path, body) {
  const r = await api.fetchApi(path, body === undefined ? undefined
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  let data = {};
  try { data = await r.json(); } catch { data = { error: `HTTP ${r.status}` }; }
  if (!r.ok && !data.error) data.error = `HTTP ${r.status}`;
  return data;
}

/* ─── add a Load node for a copied file (into the graph on screen, subgraphs included) ─── */

function _addLoader(name, kind) {
  const type = kind === "video" ? "LoadVideoC2C" : "LoadImage";
  const widgetName = kind === "video" ? "video" : "image";
  const graph = app.canvas?.graph ?? app.graph;
  const node = globalThis.LiteGraph?.createNode(type);
  if (!node || !graph) return false;
  const canvas = app.canvas;
  const el = canvas?.canvas;
  let pos = [100, 100];
  try {
    if (el && canvas.ds?.convertOffsetToCanvas) pos = canvas.ds.convertOffsetToCanvas([el.clientWidth / 2, el.clientHeight / 2]);
  } catch { /* keep the default */ }
  node.pos = [pos[0] - (node.size?.[0] ?? 300) / 2, pos[1] - 60];
  graph.add(node);
  const w = node.widgets?.find((x) => x.name === widgetName);
  if (w) {
    const values = w.options?.values;
    if (Array.isArray(values) && !values.includes(name)) values.push(name);
    w.value = name;
    try { w.callback?.(name); } catch { /* the preview refreshes on the next run */ }
  }
  canvas?.setDirty?.(true, true);
  return true;
}

let _toastHost = null;
function _toast(detail) {
  if (!_toastHost) {
    _toastHost = document.createElement("div");
    _toastHost.style.cssText = "position:fixed;right:16px;bottom:56px;z-index:var(--c2c-z-toast, 10050);display:flex;flex-direction:column;gap:6px;max-width:min(360px, calc(100vw - 32px));";
    document.body.appendChild(_toastHost);
  }
  const t = document.createElement("div");
  t.style.cssText = "background:var(--c2c-bg2);border:1px solid var(--c2c-border);border-radius:8px;padding:8px 10px;color:var(--c2c-fg);font:12px var(--c2c-font,system-ui,sans-serif);box-shadow:0 6px 20px rgba(0,0,0,.45);display:flex;gap:8px;align-items:center;";
  const label = detail.kind === "video" ? "Add Load Video" : "Add Load Image";
  t.innerHTML = `<div style="flex:1;min-width:0;"><div style="font-weight:600;">👁 New ${_esc(detail.kind || "file")}</div>
    <div style="font-size:11px;color:var(--c2c-gray400);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="${_esc(detail.path)}">${_esc(detail.name)}</div></div>
    <button style="${_SMALL}color:var(--c2c-okSoft2);">${label}</button><button style="${_SMALL}" title="Dismiss">✕</button>`;
  const [addBtn, closeBtn] = t.querySelectorAll("button");
  const drop = () => t.remove();
  addBtn.addEventListener("click", () => { _addLoader(detail.name, detail.kind); drop(); });
  closeBtn.addEventListener("click", drop);
  _toastHost.appendChild(t);
  while (_toastHost.children.length > 4) _toastHost.firstElementChild.remove();
  setTimeout(drop, 20000);
}

/* ─── panel ─── */

function _buildPanel() {
  if (_panel) return _panel;

  const overlay = document.createElement("div");
  overlay.id = "c2c-watch-folder-overlay";
  overlay.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,.65);z-index:var(--c2c-z-modal);display:flex;align-items:center;justify-content:center;";
  overlay.addEventListener("click", (e) => { if (e.target === overlay) _close(); });

  const panel = document.createElement("div");
  panel.id = "c2c-watch-folder";
  panel.style.cssText = "background:var(--c2c-bg2);border:1px solid var(--c2c-border);border-radius:10px;width:680px;max-width:calc(100vw - 32px);max-height:82vh;display:flex;flex-direction:column;overflow:hidden;box-shadow:0 12px 40px rgba(0,0,0,.6);color:var(--c2c-fg);font-family:var(--c2c-font,system-ui,sans-serif);font-size:13px;";
  panel.innerHTML = `
    <style>#c2c-watch-folder [hidden]{display:none!important}</style>
    <div style="padding:14px 16px;border-bottom:1px solid color-mix(in srgb, var(--c2c-border) 70%, transparent);display:flex;align-items:center;gap:8px;">
      <span style="font-weight:600;font-size:14px;">👁 Watch Folder</span>
      <span style="font-size:10px;color:var(--c2c-gray500);">new image / video files, checked every 2 s once they stop growing</span>
      <button id="c2c-wf-close" style="background:none;border:none;font-size:18px;cursor:pointer;color:var(--c2c-gray300);margin-left:auto;">×</button>
    </div>
    <div style="padding:12px 16px;border-bottom:1px solid color-mix(in srgb, var(--c2c-border) 50%, transparent);">
      <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:flex-end;">
        <div style="flex:2;min-width:160px;">
          <div style="${_LABEL}">Folder path</div>
          <input id="c2c-wf-path" type="text" placeholder="D:\\renders\\incoming" style="${_INPUT}">
        </div>
        <div style="flex:1;min-width:150px;">
          <div style="${_LABEL}">When a file arrives</div>
          <select id="c2c-wf-action" style="${_INPUT}background:var(--c2c-surface0);">
            <option value="notify">Log it</option>
            <option value="load_image">Copy to input + offer a Load node</option>
            <option value="run_workflow">Run a workflow on it</option>
          </select>
        </div>
        <div id="c2c-wf-src-wrap" style="flex:1;min-width:120px;" hidden>
          <div style="${_LABEL}">Workflow</div>
          <select id="c2c-wf-src" style="${_INPUT}background:var(--c2c-surface0);">
            <option value="current">Current workflow</option>
            <option value="file">API JSON file…</option>
          </select>
        </div>
        <div id="c2c-wf-workflow-wrap" style="flex:2;min-width:160px;" hidden>
          <div style="${_LABEL}">API workflow file (Workflow › Export (API))</div>
          <input id="c2c-wf-workflow" type="text" placeholder="D:\\workflows\\denoise_api.json" style="${_INPUT}">
        </div>
      </div>
      <div style="display:flex;gap:10px;align-items:center;margin-top:8px;">
        <label style="font-size:11px;color:var(--c2c-gray300);display:flex;gap:5px;align-items:center;">
          <input id="c2c-wf-recursive" type="checkbox"> Include subfolders
        </label>
        <span id="c2c-wf-hint" style="font-size:10px;color:var(--c2c-gray600);"></span>
        <button id="c2c-wf-add" style="margin-left:auto;padding:6px 14px;background:color-mix(in srgb, var(--c2c-ok) 15%, transparent);border:1px solid color-mix(in srgb, var(--c2c-ok) 30%, transparent);border-radius:5px;cursor:pointer;color:var(--c2c-okSoft2);font-size:12px;font-weight:600;white-space:nowrap;">+ Add watcher</button>
      </div>
      <div id="c2c-wf-error" style="font-size:11px;color:var(--c2c-dangerSoft);margin-top:6px;" hidden></div>
    </div>
    <div id="c2c-wf-watchers" style="padding:8px 16px;border-bottom:1px solid color-mix(in srgb, var(--c2c-border) 50%, transparent);min-height:40px;"></div>
    <div style="padding:6px 16px;font-size:10px;color:var(--c2c-gray400);font-weight:700;letter-spacing:.06em;border-bottom:1px solid color-mix(in srgb, var(--c2c-border) 40%, transparent);display:flex;align-items:center;">
      RECENT EVENTS <button id="c2c-wf-clear" style="${_SMALL}margin-left:auto;">Clear</button>
    </div>
    <div id="c2c-wf-events" style="overflow-y:auto;flex:1;padding:6px 16px;font-size:11px;"></div>
  `;
  overlay.appendChild(panel);
  document.body.appendChild(overlay);
  _panel = overlay;

  const $ = (sel) => panel.querySelector(sel);
  const HINTS = {
    notify: "",
    load_image: "Copies into input/c2c_watch so Load Image / Load Video (C2C) can read it.",
    run_workflow: "The file goes into the workflow's first Load Image (images) or Load Video (videos).",
  };
  const syncAction = () => {
    const a = $("#c2c-wf-action").value;
    $("#c2c-wf-src-wrap").hidden = a !== "run_workflow";
    $("#c2c-wf-workflow-wrap").hidden = a !== "run_workflow" || $("#c2c-wf-src").value !== "file";
    $("#c2c-wf-hint").textContent = HINTS[a] || "";
  };
  $("#c2c-wf-close").addEventListener("click", _close);
  $("#c2c-wf-add").addEventListener("click", _addWatcherUI);
  $("#c2c-wf-clear").addEventListener("click", async () => { await _call("/c2c/watchfolder/clear_events", {}); _refreshUI(); });
  $("#c2c-wf-action").addEventListener("change", syncAction);
  $("#c2c-wf-src").addEventListener("change", syncAction);
  syncAction();

  const _header = $("#c2c-wf-close").parentElement;
  attachWindowChrome(panel, { storageKey: "watch_folder", overlay, header: _header,
    titleEl: _header.querySelector("span"), minW: 520, minH: 320 });
  return overlay;
}

function _showError(msg) {
  const el = _panel?.querySelector("#c2c-wf-error");
  if (!el) return;
  el.textContent = msg || "";
  el.hidden = !msg;
}

async function _addWatcherUI() {
  const $ = (sel) => _panel.querySelector(sel);
  _showError("");
  const path = $("#c2c-wf-path").value.trim();
  const action = $("#c2c-wf-action").value;
  if (!path) { _showError("Enter a folder path."); return; }
  const body = { path, action, recursive: $("#c2c-wf-recursive").checked };
  if (action === "run_workflow") {
    if ($("#c2c-wf-src").value === "file") {
      body.workflow_path = $("#c2c-wf-workflow").value.trim();
      if (!body.workflow_path) { _showError("Enter the path of an API-format workflow file."); return; }
    } else {
      const p = await app.graphToPrompt();
      if (!p?.output || !Object.keys(p.output).length) { _showError("The current workflow has no nodes to run."); return; }
      body.workflow_json = p.output;
    }
  }
  const r = await _call("/c2c/watchfolder/add", body);
  if (r.error) { _showError(r.error); return; }
  $("#c2c-wf-path").value = "";
  _refreshUI();
}

async function _refreshUI() {
  if (!_panel || !_isOpen) return;
  let statusData, eventsData;
  try {
    [statusData, eventsData] = await Promise.all([_call("/c2c/watchfolder/status"), _call("/c2c/watchfolder/events")]);
  } catch (err) {
    console.warn("[C2C.WatchFolder] refresh failed:", err);
    return;
  }

  const watchersEl = _panel.querySelector("#c2c-wf-watchers");
  const ws = statusData.watchers || [];
  if (!ws.length) {
    watchersEl.innerHTML = `<div style="color:var(--c2c-gray600);font-size:11px;padding:6px 0;">No watchers. Files already in a folder are not events; only new ones are.</div>`;
  } else {
    const ACTION = { notify: "log", load_image: "copy + Load node", run_workflow: "run workflow" };
    watchersEl.innerHTML = ws.map((w) => `
      <div style="display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid color-mix(in srgb, var(--c2c-border) 40%, transparent);">
        <span style="width:8px;height:8px;border-radius:50%;flex-shrink:0;background:${w.running && w.state === "watching" ? "var(--c2c-okBright)" : w.running ? "var(--c2c-warn, #d9a441)" : "var(--c2c-gray500)"};" title="${_esc(w.state || "")}"></span>
        <div style="flex:1;min-width:0;">
          <div style="font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="${_esc(w.path)}">${_esc(_shortPath(w.path))}</div>
          <div style="font-size:10px;color:var(--c2c-gray500);">${_esc(ACTION[w.action] || w.action)}${w.workflow ? ` (${_esc(w.workflow)})` : ""}${w.recursive ? " · subfolders" : ""} · ${_esc(w.state || "")} · ${w.files_seen} new${w.pending ? ` · ${w.pending} still writing` : ""}</div>
          ${w.last_error ? `<div style="font-size:10px;color:var(--c2c-dangerSoft);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="${_esc(w.last_error)}">${_esc(w.last_error)}</div>` : ""}
        </div>
        <button data-remove="${_esc(w.path)}" style="${_SMALL}color:var(--c2c-dangerSoft);">Remove</button>
      </div>`).join("");
    watchersEl.querySelectorAll("[data-remove]").forEach((btn) => btn.addEventListener("click", async () => {
      await _call("/c2c/watchfolder/remove", { path: btn.dataset.remove });
      _refreshUI();
    }));
  }

  const eventsEl = _panel.querySelector("#c2c-wf-events");
  const events = (eventsData.events || []).slice(-40).reverse();
  if (!events.length) {
    eventsEl.innerHTML = `<div style="color:var(--c2c-gray600);padding:8px 0;">No events yet.</div>`;
    return;
  }
  eventsEl.innerHTML = events.map((ev, i) => {
    const name = String(ev.path || "").split(/[\\/]/).pop();
    const add = ev.type === "copied" && ev.name
      ? `<button data-add="${i}" style="${_SMALL}color:var(--c2c-okSoft2);margin-left:auto;flex-shrink:0;">${ev.kind === "video" ? "Add Load Video" : "Add Load Image"}</button>` : "";
    return `<div style="padding:3px 0;border-bottom:1px solid color-mix(in srgb, var(--c2c-border) 30%, transparent);display:flex;gap:8px;align-items:center;">
      <span style="color:var(--c2c-gray600);min-width:64px;">${new Date(ev.timestamp * 1000).toLocaleTimeString()}</span>
      <span style="color:${_EVENT_TONE[ev.type] || "var(--c2c-blue)"};min-width:56px;">${_esc(ev.type)}</span>
      <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--c2c-gray250);" title="${_esc(ev.path)}${ev.msg ? "\n" + _esc(ev.msg) : ""}">${_esc(name)}${ev.msg ? ` <span style="color:var(--c2c-gray500);">— ${_esc(ev.msg)}</span>` : ""}</span>
      ${add}
    </div>`;
  }).join("");
  eventsEl.querySelectorAll("[data-add]").forEach((b) => b.addEventListener("click", () => {
    const ev = events[Number(b.dataset.add)];
    if (ev && _addLoader(ev.name, ev.kind)) b.textContent = "Added";
  }));
}

function _open() {
  _buildPanel();
  _panel.style.display = "flex";
  _isOpen = true;
  _refreshUI();
  // Not ambient: the user opened this to watch files arrive, and ambient timers pause while a prompt runs and in
  // lite mode - exactly when the events happen.
  _refreshTimer = getRuntime().every("watchfolder.ui", 3000, () => _refreshUI(), { ambient: false });
}

function _close() {
  if (_panel) _panel.style.display = "none";
  _isOpen = false;
  if (_refreshTimer) { _refreshTimer.cancel(); _refreshTimer = null; }
}

function _buildSlot() {
  const btn = document.createElement("button");
  btn.id        = "c2c-watch-folder-btn";
  btn.className = "c2c-omnibar-slot-pill";
  btn.textContent = "👁 Watch";
  btn.title     = "Watch Folder";
  btn.style.cssText = "font-size:11px;padding:2px 7px;cursor:pointer;background:color-mix(in srgb, var(--c2c-surface1) 90%, transparent);color:var(--c2c-fg);border:1px solid color-mix(in srgb, var(--c2c-border) 24%, transparent);border-radius:10px;";
  btn.addEventListener("click", () => _isOpen ? _close() : _open());
  return btn;
}

app.registerExtension({
  name: "C2C.WatchFolder",
  // The command palette entry works where the OmniBar is not loaded (lite mode strips it).
  commands: [{ id: "C2C.watchFolder.open", label: "C2C: Watch Folder", function: () => _open() }],
  async setup() {
    // OmniBar is optional (lite mode has none): retry briefly, then stop.
    let tries = 0;
    const tryRegister = () => {
      if (window.C2COmniBar) {
        window.C2COmniBar.register({ section: "tools", id: "c2c-watch-folder", order: 40, element: _buildSlot() });
      } else if (++tries < 50) {
        setTimeout(tryRegister, 200);
      }
    };
    tryRegister();
    api.addEventListener("c2c:watchfolder:new_image", (e) => { if (e?.detail?.name) _toast(e.detail); });
    window.C2CWatchFolder = { open: _open, close: _close, addLoader: _addLoader };
  },
});
