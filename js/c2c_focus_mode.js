/**
 * c2c_focus_mode.js — Focus Mode (P2.2)
 * Dims all non-selected nodes; selected + connected neighbours glow.
 * Toggle: Ctrl+Shift+F or OmniBar tools slot.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { C } from './_c2c_theme.js';

const SETTING_ID  = "c2c.focusMode.enabled";
// DIM_OPACITY / NEIGHBOUR_OPACITY are the *remaining* visibility of a node
// after the focus veil; the black overlay alpha drawn over it is (1 - this).
// The old value 0.18 meant an ~82%-opaque black wash that made every dimmed
// node look solid-black — the reported "black shadow damaging the entire node".
// Keep dimmed nodes clearly legible so the veil can never read as "broken".
const DIM_OPACITY = 0.60;
const NEIGHBOUR_OPACITY = 0.85;

const z = { focusOverlay: 1500 };

let _enabled  = false;
let _canvas   = null;
let _origDraw = null;
let _slot     = null;

/* ─── core rendering ─── */

function _getSelected() {
  const sel = new Set();
  if (!_canvas) return sel;
  const g = _canvas.graph;
  if (!g) return sel;
  if (g.selected_nodes) {
    for (const id of Object.keys(g.selected_nodes)) sel.add(Number(id));
  }
  return sel;
}

function _getNeighbours(ids) {
  const nb = new Set();
  if (!_canvas) return nb;
  const g = _canvas.graph;
  if (!g || !g._nodes) return nb;
  for (const node of g._nodes) {
    if (!ids.has(node.id)) continue;
    if (node.inputs) {
      for (const inp of node.inputs) {
        if (inp.link != null) {
          const link = g.links[inp.link];
          if (link) nb.add(link.origin_id);
        }
      }
    }
    if (node.outputs) {
      for (const out of node.outputs) {
        if (out.links) {
          for (const lid of out.links) {
            const link = g.links[lid];
            if (link) nb.add(link.target_id);
          }
        }
      }
    }
  }
  return nb;
}

function _patchDraw() {
  if (_origDraw) return;
  if (!_canvas) return;
  _origDraw = _canvas.draw.bind(_canvas);

  _canvas.draw = function (force_canvas, force_bgcanvas) {
    if (!_enabled) {
      _origDraw(force_canvas, force_bgcanvas);
      return;
    }
    // draw normally first
    _origDraw(force_canvas, force_bgcanvas);

    const ctx   = _canvas.canvas.getContext("2d");
    const g     = _canvas.graph;
    if (!g || !g._nodes) return;

    const sel   = _getSelected();

    // With nothing selected there is no focus target, so dimming every node
    // painted the whole canvas black on idle (the reported "black shadow
    // damaging the entire node"). Only engage the veil when something is
    // actually selected.
    if (sel.size === 0) return;

    const nb    = _getNeighbours(sel);

    // dim non-selected, non-neighbour nodes
    const ds  = _canvas.ds;
    const off = ds.offset;
    const sc  = ds.scale;

    ctx.save();
    ctx.translate(off[0], off[1]);
    ctx.scale(sc, sc);

    for (const node of g._nodes) {
      if (sel.has(node.id)) continue;
      const op = nb.has(node.id) ? NEIGHBOUR_OPACITY : DIM_OPACITY;
      ctx.globalAlpha = 1 - op;
      ctx.fillStyle   = C.black;
      const size = node.size || [200, 80];
      ctx.fillRect(node.pos[0] - 2, node.pos[1] - 2, size[0] + 4, size[1] + 4);
    }
    ctx.restore();
  };
}

function _unpatchDraw() {
  if (!_origDraw || !_canvas) return;
  _canvas.draw = _origDraw;
  _origDraw    = null;
}

/* ─── toggle ─── */

function _setEnabled(v) {
  _enabled = !!v;
  app.ui.settings.setSettingValue(SETTING_ID, _enabled);
  if (_enabled) {
    _canvas = app.canvas;
    _patchDraw();
  } else {
    _unpatchDraw();
    _canvas = null;
  }
  _canvas && _canvas.setDirty(true, true);
  _updateSlot();
}

function _updateSlot() {
  if (!_slot) return;
  _slot.title   = _enabled ? "Focus mode ON (Ctrl+Shift+F)" : "Focus mode OFF (Ctrl+Shift+F)";
  _slot.setAttribute("data-active", String(_enabled));
}

/* ─── OmniBar slot ─── */

function _buildSlot() {
  const btn = document.createElement("button");
  btn.className = "c2c-omnibar-slot-pill";
  btn.id        = "c2c-focus-mode-btn";
  btn.textContent = "⊙ Focus";
  btn.setAttribute("data-active", "false");
  btn.style.cssText = `
    font-size: 11px;
    padding: 2px 7px;
    cursor: pointer;
    background: var(--c2c-pill-bg, rgba(255,255,255,.07));
    color: var(--c2c-fg, var(--c2c-gray200));
    border: 1px solid var(--c2c-border, rgba(255,255,255,.12));
    border-radius: 10px;
  `;
  btn.addEventListener("click", () => _setEnabled(!_enabled));
  _slot = btn;
  return btn;
}

/* ─── keyboard shortcut ─── */

function _onKeyDown(e) {
  if (e.ctrlKey && e.shiftKey && e.key === "F") {
    e.preventDefault();
    _setEnabled(!_enabled);
  }
}

/* ─── extension ─── */

app.registerExtension({
  name: "C2C.FocusMode",

  async setup() {
    app.ui.settings.addSetting({
      id: SETTING_ID,
      name: "C2C › Focus mode",
      type: "boolean",
      defaultValue: false,
      onChange(v) { if (_enabled !== !!v) _setEnabled(!!v); }
    });

    _enabled = app.ui.settings.getSettingValue(SETTING_ID, false);

    document.addEventListener("keydown", _onKeyDown);

    // OmniBar registration
    const tryRegister = () => {
      if (!window.C2COmniBar) return setTimeout(tryRegister, 200);
      window.C2COmniBar.register({
        section : "tools",
        id      : "c2c-focus-mode",
        order   : 25,
        element : _buildSlot(),
      });
      if (_enabled) {
        _canvas = app.canvas;
        _patchDraw();
      }
    };
    tryRegister();

    // re-patch after canvas ready. `app` does not expose `addEventListener`
    // directly — that method lives on `app.api`. Fall back to a one-shot
    // canvas-ready poll if the api channel is unavailable.
    try {
      const repatch = () => {
        if (_enabled) { _canvas = app.canvas; _patchDraw(); }
      };
      if (typeof app?.api?.addEventListener === "function") {
        app.api.addEventListener("graphChanged", repatch);
      } else {
        let tries = 0;
        const poll = () => {
          if (app?.canvas) { repatch(); return; }
          if (++tries < 40) setTimeout(poll, 250);
        };
        poll();
      }
    } catch (_err) { /* canvas not ready yet; tryRegister loop handles it */ }
  }
});
