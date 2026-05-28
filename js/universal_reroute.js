/**
 * Universal Reroute ("Dot") Node — MEC dynamic rerouter for ComfyUI
 *
 * Virtual node: strips itself from the backend prompt so it never
 * causes "Required input is missing" errors.
 *
 * Features:
 *   - Drop onto ANY connection → auto-adapts slot types
 *   - Compact circle with web-strand accents (lightweight canvas only)
 *   - Bundle drop: intercept nearby wires on placement
 *   - Right-click → "Remove Reroute (reconnect)" to dissolve
 *   - Double-click to toggle type label
 *   - Zero GPU cost — pure Canvas2D rendering
 */

import { app } from "../../scripts/app.js";
import { C } from './_c2c_theme.js';
// Optional: noodle-style helpers (separate extension). Import is lazy /
// non-fatal — if the file is missing or the extension hasn't registered
// yet, we degrade gracefully and just don't show the noodle submenu.
let NOODLE_STYLES = null;
let NOODLE_SETTING_ID = null;
import("./c2c_noodle_styles.js")
import { reportFailure as __c2cReport } from "./_c2c_report.js";
    .then((m) => { NOODLE_STYLES = m.NOODLE_STYLES; NOODLE_SETTING_ID = m.NOODLE_SETTING_ID; })
    .catch(() => { /* file absent; reroute menu still works without it */ });

const NODE_TYPE   = "UniversalRerouteMEC";
const NODE_WIDTH  = 40;
const NODE_HEIGHT = 30;
const DOT_RADIUS  = 9;
const HIT_RADIUS  = 100;

// ── Type → color (matches ComfyUI link palette) ─────────────────────
const TYPE_COLORS = {
  IMAGE:        "var(--c2c-blueSoft2)",
  LATENT:       "var(--c2c-pinkMid)",
  MASK:         "var(--c2c-okMid)",
  MODEL:        "var(--c2c-violet)",
  CLIP:         "var(--c2c-amberSoft2)",
  VAE:          "var(--c2c-cyanSoft)",
  CONDITIONING: "var(--c2c-amberMid2)",
  INT:          "var(--c2c-blueSoft)",
  FLOAT:        "var(--c2c-blueSoft)",
  STRING:       "var(--c2c-okPale2)",
  BOOLEAN:      "var(--c2c-violetSoft2)",
  COMBO:        "var(--c2c-slate450)",
  BBOX:         "var(--c2c-dangerTint2)",
  SAM_MODEL:    "var(--c2c-violet)",
  CONTROL_NET:  "var(--c2c-tealMid)",
  SEC_MODEL:    "var(--c2c-dangerSoft4)",
  SAM2MODEL:    "var(--c2c-violet)",
  "*":          "var(--c2c-gray400)",
};

function typeColor(t) { return TYPE_COLORS[t] || TYPE_COLORS["*"]; }

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app.registerExtension({
  name: "MEC.UniversalReroute",

  // ── Backend node hooks ─────────────────────────────────────────────
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;
    nodeType.prototype.onConnectInput  = () => true;
    nodeType.prototype.onConnectOutput = () => true;
  },

  // ── Per-instance setup ─────────────────────────────────────────────
  nodeCreated(node) {
    if (node.comfyClass !== NODE_TYPE) return;

    node.setSize([NODE_WIDTH, NODE_HEIGHT]);
    node.color   = "var(--c2c-panelHi)";
    node.bgcolor = "var(--c2c-panelHi)";
    node.shape   = LiteGraph.BOX_SHAPE;
    node.serialize_widgets = false;
    node.isVirtualNode = true;

    if (!node.properties) node.properties = {};
    node.properties.showLabel = false;

    // ── Slot type adaptation ─────────────────────────────────────────
    const origCC = node.onConnectionsChange;
    node.onConnectionsChange = function (side, _idx, connected, linkInfo) {
      origCC?.apply(this, arguments);
      if (!linkInfo) return;
      const g = this.graph || app.graph;
      if (!g) return;
      const link = g.links?.[linkInfo.id ?? linkInfo];
      if (!link) return;

      let resolved = null;
      if (side === LiteGraph.INPUT && connected) {
        const src = g.getNodeById(link.origin_id);
        resolved = src?.outputs?.[link.origin_slot]?.type || "*";
      } else if (side === LiteGraph.OUTPUT && connected) {
        const tgt = g.getNodeById(link.target_id);
        resolved = tgt?.inputs?.[link.target_slot]?.type || "*";
      }

      if (resolved && resolved !== "*") {
        if (this.inputs?.[0])  { this.inputs[0].type = resolved;  this.inputs[0].name = ""; }
        if (this.outputs?.[0]) { this.outputs[0].type = resolved; this.outputs[0].name = ""; }
      }

      if (!connected) {
        const hasIn  = this.inputs?.[0]?.link != null;
        const hasOut = this.outputs?.[0]?.links?.length > 0;
        if (!hasIn && !hasOut) {
          if (this.inputs?.[0])  { this.inputs[0].type = "*"; this.inputs[0].name = ""; }
          if (this.outputs?.[0]) { this.outputs[0].type = "*"; this.outputs[0].name = ""; }
        }
      }
      this.setDirtyCanvas?.(true, true);
    };

    // ── Draw: compact circle with web-strand accents ─────────────────
    node.onDrawForeground = function (ctx) {
      const t = this.inputs?.[0]?.type || this.outputs?.[0]?.type || "*";
      const c = typeColor(t);
      const cx = this.size[0] / 2;
      const cy = this.size[1] / 2;
      const r  = DOT_RADIUS;

      // Outer ring
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = C.panelDeep10;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = c;
      ctx.stroke();

      // Inner dot
      ctx.beginPath();
      ctx.arc(cx, cy, r * 0.35, 0, Math.PI * 2);
      ctx.fillStyle = c;
      ctx.fill();

      // Web strands — 6 thin lines radiating from center (very lightweight)
      ctx.save();
      ctx.globalAlpha = 0.25;
      ctx.strokeStyle = c;
      ctx.lineWidth = 0.7;
      for (let i = 0; i < 6; i++) {
        const angle = (Math.PI * 2 * i) / 6;
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(angle) * r * 0.4, cy + Math.sin(angle) * r * 0.4);
        ctx.lineTo(cx + Math.cos(angle) * r * 0.9, cy + Math.sin(angle) * r * 0.9);
        ctx.stroke();
      }
      ctx.restore();

      // Type label
      if (this.properties.showLabel && t !== "*") {
        ctx.font = "8px Inter, system-ui, sans-serif";
        ctx.fillStyle = C.subtext1;
        ctx.textAlign = "center";
        ctx.fillText(t, cx, cy + r + 11);
      }
    };

    // ── Double-click → toggle label ──────────────────────────────────
    const origDbl = node.onDblClick;
    node.onDblClick = function () {
      origDbl?.apply(this, arguments);
      this.properties.showLabel = !this.properties.showLabel;
      this.setDirtyCanvas?.(true, true);
    };

    // ── Context menu ─────────────────────────────────────────────────
    const origMenu = node.getExtraMenuOptions;
    node.getExtraMenuOptions = function (_canvas, options) {
      origMenu?.apply(this, arguments);

      // 1. Reroute-specific actions
      const base = [
        {
          content: "Remove Reroute (reconnect)",
          callback: () => dissolveReroute(this),
        },
        {
          content: this.properties.showLabel ? "Hide Type Label" : "Show Type Label",
          callback: () => {
            this.properties.showLabel = !this.properties.showLabel;
            this.setDirtyCanvas?.(true, true);
          },
        },
      ];

      // 2. Get/Set conversion (rgthree-style). Detect a registered
      //    SetNode/GetNode pair if present; otherwise show a disabled
      //    placeholder so the user knows the menu slot is reserved.
      const hasSetGet = !!(window.LiteGraph?.registered_node_types?.["SetNode"]
                       && window.LiteGraph?.registered_node_types?.["GetNode"]);
      base.push({
        content: hasSetGet
          ? "Convert to Get / Set Variable"
          : "Convert to Get / Set (install rgthree-comfy to enable)",
        disabled: !hasSetGet,
        callback: hasSetGet ? () => convertToGetSet(this) : undefined,
      });

      // 3. Noodle-style quick-switch submenu (if extension is loaded)
      if (NOODLE_STYLES && NOODLE_SETTING_ID) {
        let cur = "default";
        try { cur = app.ui.settings.getSettingValue(NOODLE_SETTING_ID, "default"); } catch (__c2cErr) { __c2cReport("universal_reroute", __c2cErr); }
        const submenu = NOODLE_STYLES.map((s) => ({
          content: `${s === cur ? "✓ " : "  "}${s}`,
          callback: () => {
            try { app.ui.settings.setSettingValue(NOODLE_SETTING_ID, s); } catch (__c2cErr) { __c2cReport("universal_reroute", __c2cErr); }
            app.canvas?.setDirty?.(true, true);
          },
        }));
        base.push({
          content: "🍝 Noodle Style",
          has_submenu: true,
          submenu: { options: submenu },
        });
      }

      // 4. Hint that all of the above also lives in Settings
      base.push({
        content: "— configurable in Settings → mec.noodle.style —",
        disabled: true,
      });

      options.unshift(...base);
    };
  },

  // ── Canvas-level setup ─────────────────────────────────────────────
  setup() {
    // Strip from prompt before execution
    const origGTP = app.graphToPrompt?.bind(app);
    if (origGTP) {
      app.graphToPrompt = async function () {
        const p = await origGTP();
        if (p?.output) {
          for (const k of Object.keys(p.output)) {
            if (p.output[k]?.class_type === NODE_TYPE) delete p.output[k];
          }
        }
        return p;
      };
    }

    // Bundle-drop on move
    const origMoved = app.canvas?.onNodeMoved?.bind(app.canvas);
    if (app.canvas) {
      app.canvas.onNodeMoved = function (node) {
        origMoved?.(node);
        if (node?.comfyClass === NODE_TYPE && !node._mecWired) tryWireOnDrop(node);
      };
    }

    // Right-click canvas → "Insert Reroute (MEC)"
    const origCanvasMenu = LGraphCanvas.prototype.getCanvasMenuOptions;
    if (origCanvasMenu) {
      LGraphCanvas.prototype.getCanvasMenuOptions = function () {
        const opts = origCanvasMenu.apply(this, arguments);
        opts.push(null, {
          content: "Insert Reroute (MEC)",
          callback: () => insertRerouteAtMouse(this),
        });
        return opts;
      };
    }
  },
});

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Convert to Get/Set (rgthree-style virtual wires)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// Only fires when rgthree-comfy's SetNode/GetNode are registered. Drops
// a Set on the source side and a Get on each target side, with a shared
// auto-generated variable name, then removes this dot.

function convertToGetSet(node) {
  const g = node.graph || app.graph;
  if (!g || !window.LiteGraph?.registered_node_types?.["SetNode"]
        || !window.LiteGraph?.registered_node_types?.["GetNode"]) {
    return;
  }
  const inLink = node.inputs?.[0]?.link;
  let srcId = null, srcSlot = null;
  if (inLink != null) {
    const lk = g.links?.[inLink];
    if (lk) { srcId = lk.origin_id; srcSlot = lk.origin_slot; }
  }
  const targets = [];
  for (const lid of (node.outputs?.[0]?.links || [])) {
    const lk = g.links?.[lid];
    if (lk) targets.push({ id: lk.target_id, slot: lk.target_slot });
  }
  const name = `var_${Math.random().toString(36).slice(2, 8)}`;

  const setNode = window.LiteGraph.createNode("SetNode");
  const getNode = window.LiteGraph.createNode("GetNode");
  if (!setNode || !getNode) return;
  setNode.pos = [node.pos[0] - 120, node.pos[1]];
  getNode.pos = [node.pos[0] + 120, node.pos[1]];
  if (setNode.widgets?.[0]) setNode.widgets[0].value = name;
  if (getNode.widgets?.[0]) getNode.widgets[0].value = name;
  g.add(setNode); g.add(getNode);

  if (srcId != null) {
    const src = g.getNodeById(srcId);
    src?.connect?.(srcSlot, setNode, 0);
  }
  for (const tgt of targets) {
    const dst = g.getNodeById(tgt.id);
    if (dst) getNode.connect(0, dst, tgt.slot);
  }
  g.remove(node);
  app.canvas?.setDirty?.(true, true);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Dissolve — reconnect source → targets, remove the dot
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function dissolveReroute(node) {
  const g = node.graph || app.graph;
  if (!g) return;

  const inLink = node.inputs?.[0]?.link;
  let srcId = null, srcSlot = null;
  if (inLink != null) {
    const lk = g.links?.[inLink];
    if (lk) { srcId = lk.origin_id; srcSlot = lk.origin_slot; }
  }

  const targets = [];
  for (const lid of (node.outputs?.[0]?.links || [])) {
    const lk = g.links?.[lid];
    if (lk) targets.push({ id: lk.target_id, slot: lk.target_slot });
  }

  node.disconnectInput(0);
  node.disconnectOutput(0);

  if (srcId != null) {
    const src = g.getNodeById(srcId);
    if (src) {
      for (const t of targets) {
        const tgt = g.getNodeById(t.id);
        if (tgt) forceConnect(g, src, srcSlot, tgt, t.slot);
      }
    }
  }
  g.remove(node);
  g.setDirtyCanvas?.(true, true);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Wire-on-drop — insert into nearby link when node is placed
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function tryWireOnDrop(node) {
  const g = app.graph;
  if (!g) return;
  node._mecWired = true;

  const gx = node.pos[0] + NODE_WIDTH / 2;
  const gy = node.pos[1] + NODE_HEIGHT / 2;
  const hits = linksNearPoint(g, gx, gy, node.id);
  if (!hits.length) return;

  const h = hits[0];
  const srcNode = g.getNodeById(h.link.origin_id);
  const tgtNode = g.getNodeById(h.link.target_id);
  if (!srcNode || !tgtNode) return;

  const srcType = srcNode.outputs?.[h.link.origin_slot]?.type || "*";
  if (node.inputs?.[0])  node.inputs[0].type = srcType;
  if (node.outputs?.[0]) node.outputs[0].type = srcType;

  g.removeLink(h.link.id);
  forceConnect(g, srcNode, h.link.origin_slot, node, 0);
  forceConnect(g, node, 0, tgtNode, h.link.target_slot);
  g.setDirtyCanvas(true, true);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Insert Reroute at mouse position
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function insertRerouteAtMouse(canvas) {
  const g = canvas.graph || app.graph;
  if (!g) return;

  const pos = canvas.canvas_mouse || canvas.last_mouse_position;
  if (!pos) return;
  const gp = canvas.convertEventToCanvasOffset({ clientX: pos[0], clientY: pos[1] });
  const gx = gp[0] || pos[0], gy = gp[1] || pos[1];

  const near = linksNearPoint(g, gx, gy, -1);
  const rr = LiteGraph.createNode(NODE_TYPE);
  if (!rr) return;
  rr.pos = [gx - NODE_WIDTH / 2, gy - NODE_HEIGHT / 2];
  g.add(rr);

  if (near.length) {
    const h = near[0];
    const sn = g.getNodeById(h.link.origin_id);
    const tn = g.getNodeById(h.link.target_id);
    if (sn && tn) {
      const sType = sn.outputs?.[h.link.origin_slot]?.type || "*";
      if (rr.inputs?.[0])  rr.inputs[0].type = sType;
      if (rr.outputs?.[0]) rr.outputs[0].type = sType;
      g.removeLink(h.link.id);
      forceConnect(g, sn, h.link.origin_slot, rr, 0);
      forceConnect(g, rr, 0, tn, h.link.target_slot);
    }
  }
  g.setDirtyCanvas?.(true, true);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Geometry helpers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function forceConnect(graph, srcNode, srcSlot, dstNode, dstSlot) {
  const sOut = srcNode.outputs?.[srcSlot];
  const dIn  = dstNode.inputs?.[dstSlot];
  if (!sOut || !dIn) return;
  const oldS = sOut.type, oldD = dIn.type;
  sOut.type = "*"; dIn.type = "*";
  srcNode.connect(srcSlot, dstNode, dstSlot);
  sOut.type = oldS; dIn.type = oldD;
}

function linksNearPoint(graph, gx, gy, excludeId) {
  const hits = [];
  for (const lid in graph.links) {
    const lk = graph.links[lid];
    if (!lk) continue;
    if (lk.origin_id === excludeId || lk.target_id === excludeId) continue;
    const sn = graph.getNodeById(lk.origin_id);
    const tn = graph.getNodeById(lk.target_id);
    if (!sn || !tn) continue;
    const sp = sn.getConnectionPos(false, lk.origin_slot);
    const tp = tn.getConnectionPos(true,  lk.target_slot);
    if (!sp || !tp) continue;
    const d = ptSegDist(gx, gy, sp[0], sp[1], tp[0], tp[1]);
    if (d <= HIT_RADIUS) hits.push({ link: lk, dist: d });
  }
  hits.sort((a, b) => a.dist - b.dist);
  return hits;
}

function ptSegDist(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const lenSq = dx * dx + dy * dy;
  if (lenSq === 0) return Math.hypot(px - ax, py - ay);
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lenSq));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}
