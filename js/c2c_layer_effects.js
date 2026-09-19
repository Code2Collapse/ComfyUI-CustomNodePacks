// c2c_layer_effects.js — the front-end for the MEC/LayerEffects family.
//
// The upstream pack these nodes were ported from (ComfyUI_LayerStyle by
// chflame163) ships no front-end for them at all, so this is written from the
// controls up rather than copied. Three things a compositor cannot do with the
// default widgets:
//
//   1. READ A COLOUR. Seven of the eight nodes take their colour as a STRING
//      holding "#FFBF30". Nobody knows what #FFBF30 looks like. Every colour
//      widget gets a live swatch and the OS colour picker, and the hex stays
//      editable for anyone matching a value from elsewhere.
//
//   2. AIM A SHADOW. Drop Shadow and Inner Shadow take distance_x and
//      distance_y as two integers, but nobody thinks in x/y - they think "the
//      key is up and to the left". The dial is angle-and-distance, which is how
//      Photoshop, Nuke and every gaffer on set describe it, and it writes the
//      two integers underneath. Drag it and they follow; type into them and the
//      dial follows. Neither is hidden, because matching a plate sometimes does
//      mean typing an exact pixel offset.
//
//   3. SEE A GRADIENT. Gradient Overlay and Gradient Map are two or three hex
//      strings, an angle and a midpoint. The ramp is drawn at the angle the
//      node will actually use, with the midpoint where it will actually land.
//
// What is deliberately NOT here: a preview of the effect itself. That would
// need the backend to write frames on every parameter change, and these are
// composite nodes - the result belongs downstream where the comp is already
// being viewed.
//
// Plain ES module, no Vue. addDOMWidget renders on both frontends. rAF-
// coalesced, chained onRemoved, and nothing touches `window` at import time.

import { app } from "../../scripts/app.js";
import {
  angleDial, colourRow, css, mountParts, normHex, widgetsOf,
} from "./_c2c_fx_controls.js";

const ST = "_c2cLayerFx";

const NODES = new Set([
  "LayerEffectDropShadowMEC",
  "LayerEffectInnerShadowMEC",
  "LayerEffectOuterGlowMEC",
  "LayerEffectInnerGlowMEC",
  "LayerEffectStrokeMEC",
  "LayerEffectColorOverlayMEC",
  "LayerEffectGradientOverlayMEC",
  "LayerEffectGradientMapMEC",
]);

/** Nodes whose shadow is aimed with distance_x / distance_y. */
const DIAL_NODES = new Set([
  "LayerEffectDropShadowMEC",
  "LayerEffectInnerShadowMEC",
]);

/** name -> the widgets that define its ramp. */
const RAMP_NODES = {
  LayerEffectGradientOverlayMEC: {
    stops: ["start_color", "end_color"],
    alphas: ["start_alpha", "end_alpha"],
    angle: "angle",
  },
  LayerEffectGradientMapMEC: {
    stops: ["start_color", "mid_color", "end_color"],
    mid: "mid_point",
  },
};

/** Which widgets on this node hold a colour. */
function colourWidgetNames(node) {
  return (node.widgets || [])
    .filter((w) => w.name === "color" || /_color$/.test(w.name))
    .map((w) => w.name);
}

// ── gradient ramp ───────────────────────────────────────────────────────────

const RAMP_H = 26;

function buildRamp(node, cfg) {
  const wrap = document.createElement("div");
  css(wrap, { padding: "2px 0" });

  const cv = document.createElement("canvas");
  cv.height = RAMP_H * 2;
  css(cv, { width: "100%", height: `${RAMP_H}px`, display: "block",
            borderRadius: "3px" });

  const cap = document.createElement("div");
  css(cap, { paddingTop: "2px", opacity: ".7", fontVariantNumeric: "tabular-nums" });

  wrap.append(cv, cap);

  const paint = () => {
    const w = widgetsOf(node);
    const width = Math.max(32, Math.round(cv.clientWidth * 2));
    if (cv.width !== width) cv.width = width;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const h = cv.height;

    // Alpha is part of the answer for Gradient Overlay, so the ramp sits on a
    // checkerboard rather than on the node's own background.
    const sq = 8;
    for (let y = 0; y < h; y += sq) {
      for (let x = 0; x < width; x += sq) {
        ctx.fillStyle = ((x / sq + y / sq) & 1) ? "#3a3a3a" : "#2b2b2b";
        ctx.fillRect(x, y, sq, sq);
      }
    }

    const stops = cfg.stops.map((n) => normHex(w[n]?.value, "#000000"));
    const alphas = (cfg.alphas || []).map((n) => Number(w[n]?.value ?? 255) / 255);
    const mid = cfg.mid ? Number(w[cfg.mid]?.value ?? 0.5) : null;

    const grad = ctx.createLinearGradient(0, 0, width, 0);
    const rgba = (hex, a) => {
      const r = parseInt(hex.slice(1, 3), 16);
      const g = parseInt(hex.slice(3, 5), 16);
      const b = parseInt(hex.slice(5, 7), 16);
      return `rgba(${r},${g},${b},${a ?? 1})`;
    };
    if (stops.length === 3 && mid !== null) {
      grad.addColorStop(0, rgba(stops[0]));
      grad.addColorStop(Math.min(0.999, Math.max(0.001, mid)), rgba(stops[1]));
      grad.addColorStop(1, rgba(stops[2]));
    } else {
      grad.addColorStop(0, rgba(stops[0], alphas[0]));
      grad.addColorStop(1, rgba(stops[stops.length - 1], alphas[1]));
    }
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, width, h);

    if (mid !== null) {
      const mx = Math.round(mid * width);
      ctx.strokeStyle = "rgba(255,255,255,0.7)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(mx, 0);
      ctx.lineTo(mx, h);
      ctx.stroke();
    }

    if (cfg.angle) {
      const deg = Number(w[cfg.angle]?.value ?? 0);
      // The ramp above reads left to right; say which way the node will lay it.
      const compass = ["→", "↗", "↑", "↖",
                       "←", "↙", "↓", "↘"];
      const idx = ((Math.round(deg / 45) % 8) + 8) % 8;
      cap.textContent =
        `${stops[0]} → ${stops[stops.length - 1]}   ` +
        `${deg.toFixed(0)}° ${compass[idx]}`;
    } else {
      cap.textContent =
        `${stops.join(" → ")}` +
        (mid !== null ? `   midpoint ${mid.toFixed(2)}` : "");
    }
  };

  return { wrap, paint, height: RAMP_H + 16 };
}

// ── assembly ────────────────────────────────────────────────────────────────

function attach(node, nodeName) {
  if (node[ST]) return node[ST];

  const parts = [];
  const bump = () => node[ST]?.invalidate?.();

  if (DIAL_NODES.has(nodeName)) {
    parts.push(angleDial(node, {
      mode: "xy", x: "distance_x", y: "distance_y",
      maxDistance: 200, zeroLabel: "centred \u2014 no offset",
    }, bump));
  }

  const rampCfg = RAMP_NODES[nodeName];
  if (rampCfg) parts.push(buildRamp(node, rampCfg));

  for (const name of colourWidgetNames(node)) {
    parts.push(colourRow(node, name, bump));
  }

  // mountParts owns the root element, the rAF-coalesced repaint, the widget
  // height and the teardown, so all three families behave identically.
  return mountParts(node, ST, parts);
}

app.registerExtension({
  name: "C2C.LayerEffects",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const name = String(nodeData?.name || "");
    if (!NODES.has(name)) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      try { attach(this, name); } catch (_e) { /* never break the node */ }
      return r;
    };

    // Values from a saved workflow land after onNodeCreated.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
      const r = onConfigure?.apply(this, args);
      this[ST]?.invalidate?.();
      return r;
    };
  },
});
