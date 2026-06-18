/**
 * c2c_workflow_diff.js — Workflow Diff Viewer (P2.1)
 * Compare two workflow JSONs side-by-side: added/removed/changed nodes,
 * changed widget values, rerouted connections.
 * Open via Ctrl+Shift+D or OmniBar.
 */
import { app } from "../../scripts/app.js";
import { attachWindowChrome } from "./_c2c_window.js";

let _panel  = null;
let _isOpen = false;
let _leftWF = null;
let _rightWF = null;

/* ─── diff engine ─── */

function _parseWorkflow(json) {
  try {
    const wf = typeof json === "string" ? JSON.parse(json) : json;
    const nodes = {};
    for (const n of (wf.nodes || [])) nodes[n.id] = n;
    return { nodes, links: wf.links || [], groups: wf.groups || [] };
  } catch {
    return null;
  }
}

function _diffWorkflows(a, b) {
  const result = {
    added   : [],
    removed : [],
    changed : [],
    linksDiff: []
  };

  // nodes
  for (const id of Object.keys(b.nodes)) {
    if (!a.nodes[id]) {
      result.added.push(b.nodes[id]);
    } else {
      const aNode = a.nodes[id];
      const bNode = b.nodes[id];
      const changes = [];

      // check widget values
      const aw = aNode.widgets_values || [];
      const bw = bNode.widgets_values || [];
      const maxW = Math.max(aw.length, bw.length);
      for (let i = 0; i < maxW; i++) {
        const av = JSON.stringify(aw[i]);
        const bv = JSON.stringify(bw[i]);
        if (av !== bv) {
          changes.push({ type: "widget", index: i, from: aw[i], to: bw[i] });
        }
      }

      // check position
      if (aNode.pos && bNode.pos &&
          (Math.abs(aNode.pos[0] - bNode.pos[0]) > 10 ||
           Math.abs(aNode.pos[1] - bNode.pos[1]) > 10)) {
        changes.push({ type: "pos", from: aNode.pos, to: bNode.pos });
      }

      // check type
      if (aNode.type !== bNode.type) {
        changes.push({ type: "type", from: aNode.type, to: bNode.type });
      }

      if (changes.length) {
        result.changed.push({ node: bNode, changes });
      }
    }
  }

  for (const id of Object.keys(a.nodes)) {
    if (!b.nodes[id]) result.removed.push(a.nodes[id]);
  }

  // links: compare counts
  if (a.links.length !== b.links.length) {
    result.linksDiff.push({ from: a.links.length, to: b.links.length });
  }

  return result;
}

/* ─── panel ─── */

function _buildPanel() {
  if (_panel) return _panel;

  const overlay = document.createElement("div");
  overlay.id = "c2c-wf-diff-overlay";
  overlay.style.cssText = `
    position: fixed; inset: 0;
    background: rgba(0,0,0,.65);
    z-index: var(--c2c-z-modal);
    display: flex;
    align-items: center;
    justify-content: center;
  `;
  overlay.addEventListener("click", (e) => { if (e.target === overlay) _close(); });

  const panel = document.createElement("div");
  panel.id = "c2c-wf-diff";
  panel.style.cssText = `
    background: var(--c2c-surface, var(--c2c-neutral950));
    border: 1px solid var(--c2c-border, rgba(255,255,255,.15));
    border-radius: 10px;
    width: 820px;
    max-height: 80vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow: 0 12px 40px rgba(0,0,0,.6);
    color: var(--c2c-fg, var(--c2c-gray100));
    font-family: var(--c2c-font, system-ui, sans-serif);
    font-size: 13px;
  `;

  panel.innerHTML = `
    <div style="padding:14px 16px;border-bottom:1px solid rgba(255,255,255,.1);
                display:flex;align-items:center;gap:8px;">
      <span style="font-weight:600;font-size:14px;">⇄ Workflow Diff</span>
      <button id="c2c-wf-diff-load-current"
        style="font-size:11px;padding:3px 9px;background:rgba(100,160,255,.15);
               border:1px solid rgba(100,160,255,.3);border-radius:5px;cursor:pointer;color:var(--c2c-blue);margin-left:auto;">
        Use Current as Left
      </button>
      <button id="c2c-wf-diff-close"
        style="background:none;border:none;font-size:18px;cursor:pointer;color:var(--c2c-gray300);">×</button>
    </div>
    <div style="display:flex;gap:10px;padding:12px 16px;border-bottom:1px solid rgba(255,255,255,.06);">
      <div style="flex:1;">
        <div style="font-size:11px;color:var(--c2c-gray400);margin-bottom:4px;">Left (base)</div>
        <textarea id="c2c-wf-diff-left" placeholder="Paste workflow JSON or use 'Use Current as Left'…"
          style="width:100%;height:80px;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.12);
                 border-radius:5px;padding:6px;color:inherit;font-size:11px;font-family:monospace;resize:vertical;">
        </textarea>
      </div>
      <div style="flex:1;">
        <div style="font-size:11px;color:var(--c2c-gray400);margin-bottom:4px;">Right (new)</div>
        <textarea id="c2c-wf-diff-right" placeholder="Paste workflow JSON to compare…"
          style="width:100%;height:80px;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.12);
                 border-radius:5px;padding:6px;color:inherit;font-size:11px;font-family:monospace;resize:vertical;">
        </textarea>
      </div>
    </div>
    <div style="padding:8px 16px;border-bottom:1px solid rgba(255,255,255,.06);display:flex;gap:8px;">
      <button id="c2c-wf-diff-run"
        style="padding:5px 16px;background:rgba(100,200,100,.18);border:1px solid rgba(100,200,100,.3);
               border-radius:5px;cursor:pointer;color:var(--c2c-ok);font-size:12px;font-weight:600;">
        ▶ Compare
      </button>
      <button id="c2c-wf-diff-save-left"
        style="padding:5px 12px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);
               border-radius:5px;cursor:pointer;color:var(--c2c-gray300);font-size:12px;">
        📁 Load File…
      </button>
      <input type="file" id="c2c-wf-diff-file-input" accept=".json" style="display:none;">
      <span id="c2c-wf-diff-summary" style="flex:1;text-align:right;font-size:11px;color:var(--c2c-gray400);padding:5px 0;"></span>
    </div>
    <div id="c2c-wf-diff-results"
      style="overflow-y:auto;flex:1;padding:12px 16px;"></div>
  `;

  overlay.appendChild(panel);
  document.body.appendChild(overlay);
  _panel = overlay;

  panel.querySelector("#c2c-wf-diff-close").addEventListener("click", _close);
  panel.querySelector("#c2c-wf-diff-run").addEventListener("click", _runDiff);
  panel.querySelector("#c2c-wf-diff-load-current").addEventListener("click", _loadCurrent);
  panel.querySelector("#c2c-wf-diff-save-left").addEventListener("click", () => {
    panel.querySelector("#c2c-wf-diff-file-input").click();
  });
  panel.querySelector("#c2c-wf-diff-file-input").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const text = await file.text();
    // load into whichever textarea is empty, preferring right
    const leftTA  = panel.querySelector("#c2c-wf-diff-left");
    const rightTA = panel.querySelector("#c2c-wf-diff-right");
    if (!rightTA.value.trim()) rightTA.value = text;
    else leftTA.value = text;
    e.target.value = "";
  });

  const _header  = panel.firstElementChild;
  const _titleEl = _header?.querySelector("span");
  attachWindowChrome(panel, {
    storageKey: "workflow_diff",
    overlay,
    header: _header,
    titleEl: _titleEl,
    shortcut: "Ctrl+Shift+D",
    minW: 560, minH: 360,
  });

  return overlay;
}

function _loadCurrent() {
  if (!app.graph) return;
  try {
    const wf = app.graph.serialize();
    const json = JSON.stringify(wf, null, 2);
    document.querySelector("#c2c-wf-diff-left").value = json;
  } catch (err) {
    console.error("[C2C.WFDiff] serialize error:", err);
  }
}

function _runDiff() {
  const leftText  = document.querySelector("#c2c-wf-diff-left")?.value?.trim();
  const rightText = document.querySelector("#c2c-wf-diff-right")?.value?.trim();
  const results   = document.getElementById("c2c-wf-diff-results");
  const summary   = document.getElementById("c2c-wf-diff-summary");
  if (!results || !summary) return;

  if (!leftText || !rightText) {
    results.innerHTML = `<p style="color:var(--c2c-dangerSoft);text-align:center;padding:20px;">Paste both workflows to compare.</p>`;
    return;
  }

  const a = _parseWorkflow(leftText);
  const b = _parseWorkflow(rightText);

  if (!a || !b) {
    results.innerHTML = `<p style="color:var(--c2c-dangerSoft);text-align:center;padding:20px;">Invalid JSON in one or both fields.</p>`;
    return;
  }

  const diff = _diffWorkflows(a, b);
  const total = diff.added.length + diff.removed.length + diff.changed.length;
  summary.textContent = total === 0 ? "No differences found" :
    `${diff.added.length} added · ${diff.removed.length} removed · ${diff.changed.length} changed`;

  if (total === 0 && diff.linksDiff.length === 0) {
    results.innerHTML = `<p style="color:var(--c2c-ok);text-align:center;padding:20px;">✓ Workflows are identical</p>`;
    return;
  }

  let html = "";

  if (diff.added.length) {
    html += `<div style="margin-bottom:12px;"><div style="font-size:11px;color:var(--c2c-ok);font-weight:600;margin-bottom:6px;">
      ➕ Added (${diff.added.length})</div>`;
    for (const n of diff.added) {
      html += `<div style="padding:5px 8px;background:rgba(0,255,0,.06);border-left:2px solid var(--c2c-okMute);
               border-radius:3px;margin-bottom:3px;font-size:12px;">
        <strong>#${n.id}</strong> — ${n.type || "(unknown)"} ${n.title ? `"${n.title}"` : ""}
      </div>`;
    }
    html += "</div>";
  }

  if (diff.removed.length) {
    html += `<div style="margin-bottom:12px;"><div style="font-size:11px;color:var(--c2c-dangerSoft);font-weight:600;margin-bottom:6px;">
      ➖ Removed (${diff.removed.length})</div>`;
    for (const n of diff.removed) {
      html += `<div style="padding:5px 8px;background:rgba(255,0,0,.06);border-left:2px solid var(--c2c-dangerStrong);
               border-radius:3px;margin-bottom:3px;font-size:12px;">
        <strong>#${n.id}</strong> — ${n.type || "(unknown)"} ${n.title ? `"${n.title}"` : ""}
      </div>`;
    }
    html += "</div>";
  }

  if (diff.changed.length) {
    html += `<div style="margin-bottom:12px;"><div style="font-size:11px;color:var(--c2c-peach);font-weight:600;margin-bottom:6px;">
      ✎ Changed (${diff.changed.length})</div>`;
    for (const { node, changes } of diff.changed) {
      html += `<div style="padding:5px 8px;background:rgba(255,165,0,.06);border-left:2px solid var(--c2c-peach);
               border-radius:3px;margin-bottom:4px;font-size:12px;">
        <strong>#${node.id}</strong> — ${node.type || "(unknown)"} ${node.title ? `"${node.title}"` : ""}
        <ul style="margin:4px 0 0 12px;padding:0;list-style:disc;">
          ${changes.map(c => {
            if (c.type === "widget") return `<li>Widget[${c.index}]: <code style="color:var(--c2c-peach)">${JSON.stringify(c.from)}</code> → <code style="color:var(--c2c-blue)">${JSON.stringify(c.to)}</code></li>`;
            if (c.type === "pos")    return `<li>Position moved</li>`;
            if (c.type === "type")   return `<li>Type: ${c.from} → ${c.to}</li>`;
            return `<li>${c.type}</li>`;
          }).join("")}
        </ul>
      </div>`;
    }
    html += "</div>";
  }

  if (diff.linksDiff.length) {
    html += `<div style="margin-bottom:12px;"><div style="font-size:11px;color:var(--c2c-blue);font-weight:600;margin-bottom:6px;">
      ⟵⟶ Links</div>
      <div style="padding:5px 8px;background:rgba(100,160,255,.06);border-left:2px solid var(--c2c-overlay0);
                  border-radius:3px;font-size:12px;">
        Count: ${diff.linksDiff[0].from} → ${diff.linksDiff[0].to}
      </div></div>`;
  }

  results.innerHTML = html;
}

/* ─── open / close ─── */

function _open() {
  _buildPanel();
  _panel.style.display = "flex";
  _isOpen = true;
}

function _close() {
  if (_panel) _panel.style.display = "none";
  _isOpen = false;
}

/* ─── keyboard ─── */

document.addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.shiftKey && e.key === "D") {
    e.preventDefault();
    _isOpen ? _close() : _open();
  }
});

/* ─── OmniBar slot ─── */

function _buildSlot() {
  const btn = document.createElement("button");
  btn.id        = "c2c-workflow-diff-btn";
  btn.className = "c2c-omnibar-slot-pill";
  btn.textContent = "⇄ Diff";
  btn.title     = "Workflow Diff (Ctrl+Shift+D)";
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
  return btn;
}

/* ─── extension ─── */

app.registerExtension({
  name: "C2C.WorkflowDiff",

  async setup() {
    const tryRegister = () => {
      if (!window.C2COmniBar) return setTimeout(tryRegister, 200);
      window.C2COmniBar.register({
        section : "tools",
        id      : "c2c-workflow-diff",
        order   : 35,
        element : _buildSlot(),
      });
    };
    tryRegister();

    window.C2CWorkflowDiff = { open: _open, close: _close };
  }
});
