/**
 * c2c_keyboard_editor.js — Keyboard Shortcut Editor (P2.2)
 * Remap every C2C shortcut. Ctrl+K search palette.
 * Persists via ComfyUI settings API.
 */
import { app } from "../../scripts/app.js";
import { attachWindowChrome } from "./_c2c_window.js";

const SETTING_PREFIX = "c2c.keymap.";

// Default C2C shortcuts — id: { action, keys[], description }
const DEFAULT_KEYMAP = {
  "focusMode":        { action: "C2C.FocusMode.toggle",    keys: ["Ctrl+Shift+F"], description: "Toggle Focus Mode" },
  "workflowFind":     { action: "C2C.WorkflowFind.open",    keys: ["Ctrl+F"],       description: "Find Nodes in Workflow" },
  "promptWizard":     { action: "C2C.PromptWizard.open",    keys: ["Ctrl+Shift+P"], description: "Open Prompt Wizard" },
  "surpriseMe":       { action: "C2C.SurpriseMe.trigger",   keys: ["Ctrl+Shift+S"], description: "Surprise Me" },
  "nodeExplain":      { action: "C2C.NodeExplain.explain",  keys: [],               description: "Explain Hovered Node" },
  "commandPalette":   { action: "C2C.CommandPalette.open",  keys: ["Ctrl+K"],       description: "Open Command Palette" },
  "autoLayout":       { action: "C2C.AutoLayout.run",       keys: ["Ctrl+Alt+L"],   description: "Auto-Layout Graph" },
  "undoPanel":        { action: "C2C.UndoPanel.open",       keys: ["Ctrl+Shift+Z"], description: "Open Undo Panel" },
  "workflowDiff":     { action: "C2C.WorkflowDiff.open",    keys: ["Ctrl+Shift+D"], description: "Workflow Diff" },
  "modelBrowser":     { action: "C2C.ModelBrowser.open",    keys: ["Ctrl+Shift+M"], description: "Model Browser" },
  "tensorInspector":  { action: "C2C.TensorInspector.open", keys: ["Ctrl+Shift+T"], description: "Tensor Inspector" },
  "whatsWired":       { action: "C2C.WhatsWired.open",      keys: ["Ctrl+Shift+W"], description: "What's Wired" },
  "flamegraph":       { action: "C2C.Flamegraph.toggle",    keys: ["Ctrl+Shift+G"], description: "Toggle Flamegraph" },
  "abCanvas":         { action: "C2C.ABCanvas.toggle",      keys: ["Ctrl+Shift+A"], description: "Toggle A/B Canvas" },
};

let _keymap   = {};
let _panel    = null;
let _isOpen   = false;
let _filter   = "";
let _slot     = null;

/* ─── persistence ─── */

function _loadKeymap() {
  _keymap = {};
  for (const [id, def] of Object.entries(DEFAULT_KEYMAP)) {
    const saved = app.ui.settings.getSettingValue(SETTING_PREFIX + id, null);
    _keymap[id] = saved !== null ? { ...def, keys: JSON.parse(saved) } : { ...def };
  }
}

function _saveEntry(id, keys) {
  _keymap[id].keys = keys;
  app.ui.settings.setSettingValue(SETTING_PREFIX + id, JSON.stringify(keys));
  _broadcastRemap(id, keys);
}

function _broadcastRemap(id, keys) {
  window.dispatchEvent(new CustomEvent("c2c:keymapChanged", {
    detail: { id, action: _keymap[id].action, keys }
  }));
}

/* ─── key capture ─── */

function _keysToString(e) {
  const parts = [];
  if (e.ctrlKey  || e.metaKey) parts.push("Ctrl");
  if (e.altKey)   parts.push("Alt");
  if (e.shiftKey) parts.push("Shift");
  const k = e.key;
  if (!["Control","Alt","Shift","Meta"].includes(k)) parts.push(k);
  return parts.join("+");
}

/* ─── panel UI ─── */

function _buildPanel() {
  if (_panel) return _panel;

  const overlay = document.createElement("div");
  overlay.id = "c2c-keyboard-editor-overlay";
  overlay.style.cssText = `
    position: fixed; inset: 0;
    background: rgba(0,0,0,.6);
    z-index: var(--c2c-z-modal);
    display: flex;
    align-items: center;
    justify-content: center;
  `;
  overlay.addEventListener("click", (e) => { if (e.target === overlay) _close(); });

  const panel = document.createElement("div");
  panel.id = "c2c-keyboard-editor";
  panel.style.cssText = `
    background: var(--c2c-surface, var(--c2c-neutral950));
    border: 1px solid var(--c2c-border, rgba(255,255,255,.15));
    border-radius: 10px;
    width: 560px;
    max-height: 70vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow: 0 12px 40px rgba(0,0,0,.6);
    color: var(--c2c-fg, var(--c2c-gray100));
    font-family: var(--c2c-font, system-ui, sans-serif);
    font-size: 13px;
  `;

  // header
  const header = document.createElement("div");
  header.style.cssText = `
    padding: 14px 16px 10px;
    border-bottom: 1px solid var(--c2c-border, rgba(255,255,255,.1));
    display: flex;
    align-items: center;
    gap: 10px;
  `;
  header.innerHTML = `
    <span style="font-weight:600;font-size:14px;">⌨ Keyboard Shortcuts</span>
    <input id="c2c-key-filter" type="text" placeholder="Search…"
      style="flex:1;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);
             border-radius:5px;padding:4px 8px;color:inherit;font-size:12px;">
    <button id="c2c-key-reset-all"
      style="font-size:11px;padding:3px 8px;background:rgba(255,0,0,.12);border:1px solid rgba(255,80,80,.25);
             border-radius:5px;cursor:pointer;color:var(--c2c-dangerSoft);">Reset All</button>
    <button id="c2c-key-close"
      style="background:none;border:none;font-size:18px;cursor:pointer;color:var(--c2c-gray300);padding:0 4px;">×</button>
  `;

  // list
  const list = document.createElement("div");
  list.id = "c2c-key-list";
  list.style.cssText = `overflow-y:auto;flex:1;padding:8px 0;`;

  panel.appendChild(header);
  panel.appendChild(list);
  overlay.appendChild(panel);
  document.body.appendChild(overlay);

  overlay.querySelector("#c2c-key-close").addEventListener("click", _close);
  overlay.querySelector("#c2c-key-filter").addEventListener("input", (e) => {
    _filter = e.target.value.toLowerCase();
    _renderList();
  });
  overlay.querySelector("#c2c-key-reset-all").addEventListener("click", () => {
    for (const id of Object.keys(_keymap)) {
      _keymap[id].keys = [...DEFAULT_KEYMAP[id].keys];
      app.ui.settings.setSettingValue(SETTING_PREFIX + id, JSON.stringify(_keymap[id].keys));
      _broadcastRemap(id, _keymap[id].keys);
    }
    _renderList();
  });

  _panel = overlay;

  const _titleEl = header.querySelector("span");
  attachWindowChrome(panel, {
    storageKey: "keyboard_editor",
    overlay,
    header,
    titleEl: _titleEl,
    minW: 420, minH: 280,
  });

  return overlay;
}

function _renderList() {
  const list = document.getElementById("c2c-key-list");
  if (!list) return;
  list.innerHTML = "";

  for (const [id, entry] of Object.entries(_keymap)) {
    if (_filter && !entry.description.toLowerCase().includes(_filter) &&
        !entry.keys.join(",").toLowerCase().includes(_filter)) continue;

    const row = document.createElement("div");
    row.style.cssText = `
      display: flex;
      align-items: center;
      padding: 7px 16px;
      gap: 10px;
      border-bottom: 1px solid rgba(255,255,255,.04);
    `;
    row.innerHTML = `
      <span style="flex:1;">${entry.description}</span>
      <div id="c2c-key-badge-${id}" style="display:flex;gap:4px;"></div>
      <button data-id="${id}"
        style="font-size:10px;padding:2px 7px;background:rgba(255,255,255,.07);
               border:1px solid rgba(255,255,255,.12);border-radius:4px;cursor:pointer;color:var(--c2c-gray300);">
        Edit
      </button>
      <button data-reset="${id}"
        style="font-size:10px;padding:2px 7px;background:rgba(255,80,80,.08);
               border:1px solid rgba(255,80,80,.18);border-radius:4px;cursor:pointer;color:var(--c2c-dangerSoft);">
        ↺
      </button>
    `;
    list.appendChild(row);
    _renderBadges(id);

    row.querySelector(`[data-id="${id}"]`).addEventListener("click", () => _captureKey(id));
    row.querySelector(`[data-reset="${id}"]`).addEventListener("click", () => {
      _saveEntry(id, [...DEFAULT_KEYMAP[id].keys]);
      _renderBadges(id);
    });
  }
}

function _renderBadges(id) {
  const badge = document.getElementById(`c2c-key-badge-${id}`);
  if (!badge) return;
  badge.innerHTML = "";
  for (const k of _keymap[id].keys) {
    const span = document.createElement("kbd");
    span.textContent = k;
    span.style.cssText = `
      background: rgba(255,255,255,.1);
      border: 1px solid rgba(255,255,255,.2);
      border-radius: 4px;
      padding: 1px 6px;
      font-size: 11px;
      white-space: nowrap;
    `;
    badge.appendChild(span);
  }
  if (_keymap[id].keys.length === 0) {
    const none = document.createElement("span");
    none.textContent = "(none)";
    none.style.color = "var(--c2c-gray500)";
    none.style.fontSize = "11px";
    badge.appendChild(none);
  }
}

function _captureKey(id) {
  const row = document.querySelector(`[data-id="${id}"]`);
  if (!row) return;
  row.textContent = "Press key…";
  row.style.color = "var(--c2c-accent, var(--c2c-accentSoft))";

  const handler = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.key === "Escape") {
      row.textContent = "Edit";
      row.style.color = "";
      document.removeEventListener("keydown", handler, true);
      return;
    }
    const combo = _keysToString(e);
    if (!["Ctrl","Alt","Shift"].includes(combo)) {
      _saveEntry(id, combo ? [combo] : []);
      _renderBadges(id);
      row.textContent = "Edit";
      row.style.color = "";
      document.removeEventListener("keydown", handler, true);
    }
  };
  document.addEventListener("keydown", handler, true);
}

/* ─── open / close ─── */

function _open() {
  _loadKeymap();
  _buildPanel();
  _panel.style.display = "flex";
  _isOpen = true;
  _renderList();
  setTimeout(() => document.getElementById("c2c-key-filter")?.focus(), 50);
}

function _close() {
  if (_panel) _panel.style.display = "none";
  _isOpen = false;
}

/* ─── global keyboard listener ─── */

document.addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.key === "k" && !e.shiftKey) {
    // Ctrl+K → Command Palette (handled elsewhere) but Ctrl+Shift+K → keyboard editor
  }
  if (e.ctrlKey && e.shiftKey && e.key === "K") {
    e.preventDefault();
    _isOpen ? _close() : _open();
  }
});

/* ─── OmniBar slot ─── */

function _buildSlot() {
  const btn = document.createElement("button");
  btn.id        = "c2c-keyboard-editor-btn";
  btn.className = "c2c-omnibar-slot-pill";
  btn.textContent = "⌨ Keys";
  btn.title     = "Keyboard Editor (Ctrl+Shift+K)";
  btn.style.cssText = `
    font-size: 11px;
    padding: 2px 7px;
    cursor: pointer;
    background: var(--c2c-pill-bg, rgba(255,255,255,.07));
    color: var(--c2c-fg, var(--c2c-gray200));
    border: 1px solid var(--c2c-border, rgba(255,255,255,.12));
    border-radius: 10px;
  `;
  btn.addEventListener("click", () => _isOpen ? _close() : _open());
  _slot = btn;
  return btn;
}

/* ─── extension ─── */

app.registerExtension({
  name: "C2C.KeyboardEditor",

  async setup() {
    _loadKeymap();

    const tryRegister = () => {
      if (!window.C2COmniBar) return setTimeout(tryRegister, 200);
      window.C2COmniBar.register({
        section : "tools",
        id      : "c2c-keyboard-editor",
        order   : 50,
        element : _buildSlot(),
      });
    };
    tryRegister();

    // expose open/close for other modules
    window.C2CKeyboardEditor = { open: _open, close: _close };
  }
});
