// c2c_mask_toolkit.js — front-end previews for Mask Tools (MaskToolsMEC).
//
// One merged node (L7.65 P03): a mode combo picks colour / luma / difference keys,
// grade, grow-shrink, combine, gradient, grain, motion blur, edge spread, or shuffle.
// Widget visibility per mode is handled in c2c_mode_widgets.js — not duplicated here.
//
// Five modes get custom previews below; each is shown only while mode matches (mountParts
// visible()). Edge spread deliberately has nothing — its result depends on the plate.
//
//   key: luma        Blend If split slider (low / high / soft ends on the channel ramp).
//   gradient         Shape preview for type / angle / centre / start / end.
//   key: colour      Colour row + tolerance band strip.
//   motion blur      Polar angle dial (same control as shadow nodes).
//   grain            Live seeded grain tile.
//
// Plain ES module, no Vue. Shared controls in _c2c_fx_controls.js.

import { app } from "../../scripts/app.js";
import { C } from "./_c2c_theme.js";
import {
  angleDial, checkerCss, colourRow, css, hexToRgb, mountParts, widgetsOf,
} from "./_c2c_fx_controls.js";

const ST = "_c2cMaskKit";
const NODE = "MaskToolsMEC";

const MODE = {
  colour: "key: colour",
  luma: "key: luma",
  gradient: "gradient",
  grain: "grain",
  motionBlur: "motion blur",
};

/** Logical param -> ComfyUI widget name (matches mask_tool_widget in nodes.py). */
const W = {
  colour: {
    color: "colour_key_color",
    colorspace: "colour_key_colorspace",
    tolerance: "colour_key_tolerance",
    soft_falloff: "colour_key_soft_falloff",
  },
  luma: {
    mode: "luma_key_mode",
    low: "luma_key_low",
    high: "luma_key_high",
    gamma: "luma_key_gamma",
    invert: "luma_key_invert",
    channel: "luma_key_channel",
    low_soft: "luma_key_low_soft",
    high_soft: "luma_key_high_soft",
  },
  gradient: {
    width: "gradient_width",
    height: "gradient_height",
    gradient_type: "gradient_type",
    angle: "gradient_angle",
    center_x: "gradient_center_x",
    center_y: "gradient_center_y",
    start: "gradient_start",
    end: "gradient_end",
  },
  grain: {
    amount: "grain_amount",
    seed: "grain_seed",
    grain_size: "grain_size",
    invert: "grain_invert",
  },
  motionBlur: {
    angle: "motion_blur_angle",
    distance: "motion_blur_distance",
    invert: "motion_blur_invert",
  },
};

function wOf(node, map) {
  const all = widgetsOf(node);
  const get = (key) => all[map[key]];
  return { get, all };
}

// ── Blend If split slider ───────────────────────────────────────────────────

const STRIP_H = 46;

function blendIfStrip(node, map, onChange) {
  const wrap = document.createElement("div");
  css(wrap, { padding: "2px 0" });

  const cv = document.createElement("canvas");
  cv.height = STRIP_H * 2;
  css(cv, { width: "100%", height: `${STRIP_H}px`, display: "block",
            borderRadius: "3px", cursor: "ew-resize" });
  cv.title = "Drag a handle. The outer pair are the soft ends — how far the " +
             "key ramps in beyond the range.";

  const cap = document.createElement("div");
  css(cap, { paddingTop: "2px", opacity: ".72", whiteSpace: "pre-line",
             lineHeight: "1.4", fontVariantNumeric: "tabular-nums" });

  wrap.append(cv, cap);

  /** The four handle positions in 0..1, left to right. */
  const handles = () => {
    const { get } = wOf(node, map);
    const low = Number(get("low")?.value ?? 0);
    const high = Number(get("high")?.value ?? 1);
    const ls = Number(get("low_soft")?.value ?? 0);
    const hs = Number(get("high_soft")?.value ?? 0);
    return [
      { key: "lowSoft", v: Math.max(0, low - ls) },
      { key: "low", v: low },
      { key: "high", v: high },
      { key: "highSoft", v: Math.min(1, high + hs) },
    ];
  };

  const setFromHandle = (key, v) => {
    const { get, all } = wOf(node, map);
    const cur = handles();
    const clamp01 = (x) => Math.min(1, Math.max(0, x));
    const set = (logical, val) => {
      const wi = all[map[logical]];
      if (!wi) return;
      wi.value = Math.round(clamp01(val) * 1000) / 1000;
      wi.callback?.(wi.value);
    };
    const low = Number(get("low")?.value ?? 0);
    const high = Number(get("high")?.value ?? 1);
    if (key === "low") {
      const nv = Math.min(clamp01(v), high);
      set("low", nv);
      set("low_soft", Math.max(0, nv - cur[0].v));
    } else if (key === "high") {
      const nv = Math.max(clamp01(v), low);
      set("high", nv);
      set("high_soft", Math.max(0, cur[3].v - nv));
    } else if (key === "lowSoft") {
      set("low_soft", Math.max(0, low - clamp01(v)));
    } else {
      set("high_soft", Math.max(0, clamp01(v) - high));
    }
    node.setDirtyCanvas(true, true);
  };

  /** What the key actually returns for a given input value, 0..1. */
  const response = (x) => {
    const { get } = wOf(node, map);
    const [a, b, c, d] = handles().map((h) => h.v);
    const gamma = Math.max(0.01, Number(get("gamma")?.value ?? 1));
    const invert = !!get("invert")?.value;
    let v;
    if (x <= a) v = 0;
    else if (x >= d) v = 0;
    else if (x < b) v = (x - a) / Math.max(b - a, 1e-6);
    else if (x <= c) v = 1;
    else v = 1 - (x - c) / Math.max(d - c, 1e-6);
    v = Math.pow(Math.min(1, Math.max(0, v)), 1 / gamma);
    return invert ? 1 - v : v;
  };

  const paint = () => {
    const width = Math.max(64, Math.round(cv.clientWidth * 2));
    if (cv.width !== width) cv.width = width;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const h = cv.height;
    const { get } = wOf(node, map);
    const channel = String(get("channel")?.value ?? "luma");

    const grad = ctx.createLinearGradient(0, 0, width, 0);
    const ends = {
      luma: ["#000000", "#ffffff"],
      red: ["#000000", "#ff0000"],
      green: ["#000000", "#00ff00"],
      blue: ["#000000", "#0000ff"],
      saturation: ["#808080", "#ff0000"],
      value: ["#000000", "#ffffff"],
      L: ["#000000", "#ffffff"],
      a: ["#2f7f2f", "#c02f6f"],
      b: ["#2f4fc0", "#c0b02f"],
    }[channel];
    if (channel === "hue") {
      for (let i = 0; i <= 6; i++) {
        grad.addColorStop(i / 6, `hsl(${i * 60},100%,50%)`);
      }
    } else {
      grad.addColorStop(0, (ends || ["#000000", "#ffffff"])[0]);
      grad.addColorStop(1, (ends || ["#000000", "#ffffff"])[1]);
    }
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, width, h);

    ctx.strokeStyle = "rgba(255,255,255,0.92)";
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (let px = 0; px <= width; px += 2) {
      const y = h - response(px / width) * (h - 6) - 3;
      px === 0 ? ctx.moveTo(px, y) : ctx.lineTo(px, y);
    }
    ctx.stroke();

    const hs = handles();
    for (let i = 0; i < hs.length; i++) {
      const x = Math.round(hs[i].v * width);
      const outer = i === 0 || i === 3;
      ctx.globalAlpha = outer ? 0.85 : 1;
      ctx.fillStyle = C.amberMid;
      ctx.beginPath();
      ctx.moveTo(x, outer ? 10 : 0);
      ctx.lineTo(x - 7, outer ? 0 : -1);
      ctx.lineTo(x + 7, outer ? 0 : -1);
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = outer ? 0.55 : 1;
      ctx.strokeStyle = C.amberMid;
      ctx.lineWidth = outer ? 2 : 3;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    const keyMode = String(get("mode")?.value ?? "custom");
    const [a, b, c, d] = hs.map((x) => x.v);
    cap.textContent =
      `${channel}   keep ${b.toFixed(2)}–${c.toFixed(2)}` +
      `   soft ${(b - a).toFixed(2)} / ${(d - c).toFixed(2)}\n` +
      (keyMode === "custom"
        ? "custom — these handles are the key"
        : `mode "${keyMode}" overrides low/high; switch to custom to use them`);
  };

  let grabbed = null;
  const pick = (e) => {
    const r = cv.getBoundingClientRect();
    const x = (e.clientX - r.left) / Math.max(1, r.width);
    let best = null;
    for (const hd of handles()) {
      const d = Math.abs(hd.v - x);
      if (!best || d < best.d) best = { key: hd.key, d };
    }
    return { key: best.key, x };
  };
  cv.onpointerdown = (e) => {
    cv.setPointerCapture(e.pointerId);
    const p = pick(e);
    grabbed = p.key;
    setFromHandle(grabbed, p.x);
    paint();
    onChange?.();
  };
  cv.onpointermove = (e) => {
    if (!grabbed) return;
    const r = cv.getBoundingClientRect();
    setFromHandle(grabbed, (e.clientX - r.left) / Math.max(1, r.width));
    paint();
    onChange?.();
  };
  const up = (e) => {
    if (!grabbed) return;
    grabbed = null;
    try { cv.releasePointerCapture(e.pointerId); } catch (_e) { /* gone */ }
  };
  cv.onpointerup = up;
  cv.onpointercancel = up;

  return { wrap, paint, height: STRIP_H + 30 };
}

// ── gradient shape preview ──────────────────────────────────────────────────

const SHAPE_PX = 84;

function gradientShape(node, map) {
  const wrap = document.createElement("div");
  css(wrap, { display: "flex", alignItems: "center", gap: "8px", padding: "2px 0" });

  const cv = document.createElement("canvas");
  cv.width = SHAPE_PX * 2;
  cv.height = SHAPE_PX * 2;
  css(cv, { width: `${SHAPE_PX}px`, height: `${SHAPE_PX}px`, flex: "0 0 auto",
            borderRadius: "3px", ...checkerCss() });

  const cap = document.createElement("div");
  css(cap, { flex: "1 1 auto", minWidth: "0", whiteSpace: "pre-line",
             lineHeight: "1.45", opacity: ".8",
             fontVariantNumeric: "tabular-nums" });

  wrap.append(cv, cap);

  const paint = () => {
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const s = cv.width;
    const { get } = wOf(node, map);
    const type = String(get("gradient_type")?.value ?? "linear");
    const angle = Number(get("angle")?.value ?? 0);
    const cx = Number(get("center_x")?.value ?? 0.5);
    const cy = Number(get("center_y")?.value ?? 0.5);
    const start = Number(get("start")?.value ?? 0);
    const end = Number(get("end")?.value ?? 1);

    ctx.clearRect(0, 0, s, s);
    const img = ctx.createImageData(s, s);
    const rad = (-angle * Math.PI) / 180;
    const dx = Math.cos(rad);
    const dy = Math.sin(rad);
    const span = Math.max(1e-6, end - start);

    for (let y = 0; y < s; y++) {
      for (let x = 0; x < s; x++) {
        const u = x / (s - 1);
        const v = y / (s - 1);
        let t;
        if (type === "radial") {
          t = Math.hypot(u - cx, v - cy) * 2;
        } else if (type === "angular") {
          t = (Math.atan2(v - cy, u - cx) / (Math.PI * 2)) + 0.5;
          t = (t + angle / 360) % 1;
        } else {
          t = (u - cx) * dx + (v - cy) * dy + 0.5;
        }
        const g = Math.min(1, Math.max(0, (t - start) / span));
        const o = (y * s + x) * 4;
        const b = Math.round(g * 255);
        img.data[o] = img.data[o + 1] = img.data[o + 2] = b;
        img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);

    if (type !== "linear") {
      ctx.strokeStyle = C.amberMid;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(cx * s, cy * s, 7, 0, Math.PI * 2);
      ctx.stroke();
    }

    const sizeW = get("width")?.value ?? "?";
    const sizeH = get("height")?.value ?? "?";
    const linked = node.inputs?.some(
      (i) => i.name === "size_as" && i.link != null);
    cap.textContent =
      `${type}   ${type === "linear" || type === "angular"
        ? angle.toFixed(0) + "°" : "centre " + cx.toFixed(2) + ", " + cy.toFixed(2)}\n` +
      `ramp ${start.toFixed(2)} → ${end.toFixed(2)}\n` +
      (linked
        ? "size from the connected image — width/height ignored"
        : `${sizeW} × ${sizeH}`);
  };

  return { wrap, paint, height: SHAPE_PX + 4 };
}

// ── grain tile ──────────────────────────────────────────────────────────────

const GRAIN_PX = 64;

function grainTile(node, map) {
  const wrap = document.createElement("div");
  css(wrap, { display: "flex", alignItems: "center", gap: "8px", padding: "2px 0" });

  const cv = document.createElement("canvas");
  cv.width = GRAIN_PX * 2;
  cv.height = GRAIN_PX * 2;
  css(cv, { width: `${GRAIN_PX}px`, height: `${GRAIN_PX}px`, flex: "0 0 auto",
            borderRadius: "3px" });

  const cap = document.createElement("div");
  css(cap, { flex: "1 1 auto", minWidth: "0", whiteSpace: "pre-line",
             lineHeight: "1.45", opacity: ".8" });

  wrap.append(cv, cap);

  const rand = (seed) => {
    let s = (seed >>> 0) || 1;
    return () => {
      s ^= s << 13; s >>>= 0;
      s ^= s >> 17;
      s ^= s << 5; s >>>= 0;
      return s / 4294967296;
    };
  };

  const paint = () => {
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const { get } = wOf(node, map);
    const amount = Number(get("amount")?.value ?? 0);
    const size = Math.max(1, Number(get("grain_size")?.value ?? 1));
    const seed = Number(get("seed")?.value ?? 0);
    const invert = !!get("invert")?.value;

    const s = cv.width;
    const img = ctx.createImageData(s, s);
    const next = rand(seed + 1);
    const cells = Math.max(1, Math.round(s / (size * 2)));
    const grid = new Float32Array(cells * cells);
    for (let i = 0; i < grid.length; i++) grid[i] = next();

    const base = invert ? 0.35 : 0.65;
    for (let y = 0; y < s; y++) {
      const gy = Math.min(cells - 1, Math.floor((y / s) * cells));
      for (let x = 0; x < s; x++) {
        const gx = Math.min(cells - 1, Math.floor((x / s) * cells));
        const n = grid[gy * cells + gx] - 0.5;
        const v = Math.min(1, Math.max(0, base + n * amount));
        const o = (y * s + x) * 4;
        const b = Math.round(v * 255);
        img.data[o] = img.data[o + 1] = img.data[o + 2] = b;
        img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);

    cap.textContent =
      `amount ${amount.toFixed(2)}   size ${size}\n` +
      `seed ${seed}\n` +
      (amount === 0 ? "no grain — the matte passes through"
                    : "held still by the seed, so it will not crawl");
  };

  return { wrap, paint, height: GRAIN_PX + 4 };
}

/** The tolerance band, as a strip from the picked colour outward. */
function tolerancePreview(node, map) {
  const wrap = document.createElement("div");
  css(wrap, { padding: "2px 0" });

  const cv = document.createElement("canvas");
  cv.height = 36;
  css(cv, { width: "100%", height: "18px", display: "block", borderRadius: "3px" });

  const cap = document.createElement("div");
  css(cap, { paddingTop: "2px", opacity: ".72", whiteSpace: "pre-line",
             lineHeight: "1.4" });

  wrap.append(cv, cap);

  const paint = () => {
    const width = Math.max(64, Math.round(cv.clientWidth * 2));
    if (cv.width !== width) cv.width = width;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const { get } = wOf(node, map);
    const [r, g, b] = hexToRgb(get("color")?.value);
    const tol = Number(get("tolerance")?.value ?? 0);
    const soft = Number(get("soft_falloff")?.value ?? 0);
    const space = String(get("colorspace")?.value ?? "rgb");

    const grad = ctx.createLinearGradient(0, 0, width, 0);
    grad.addColorStop(0, `rgb(${r},${g},${b})`);
    grad.addColorStop(1, C.gray500);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, width, cv.height);

    const keep = Math.min(1, Math.max(0, tol));
    const feather = Math.min(1, Math.max(0, tol + soft));
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(Math.round(feather * width), 0,
                 width - Math.round(feather * width), cv.height);
    ctx.strokeStyle = C.white;
    ctx.lineWidth = 2;
    for (const [x, dash] of [[keep, []], [feather, [4, 4]]]) {
      ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.moveTo(Math.round(x * width), 0);
      ctx.lineTo(Math.round(x * width), cv.height);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    cap.textContent =
      `${space.toUpperCase()}   solid to ${keep.toFixed(2)}, ` +
      `feathered to ${feather.toFixed(2)}` +
      (space === "rgb"
        ? "\nRGB distance calls a dark navy and a dark brown similar — try LAB"
        : "");
  };

  return { wrap, paint, height: 36 };
}

// ── assembly ────────────────────────────────────────────────────────────────

function attachMaskTools(node) {
  const parts = [];
  const bump = () => node[ST]?.invalidate?.();
  const currentMode = () => String(widgetsOf(node).mode?.value ?? "");
  const vis = (label) => () => currentMode() === label;

  const lumaStrip = blendIfStrip(node, W.luma, bump);
  lumaStrip.visible = vis(MODE.luma);
  parts.push(lumaStrip);

  const grad = gradientShape(node, W.gradient);
  grad.visible = vis(MODE.gradient);
  parts.push(grad);

  const grain = grainTile(node, W.grain);
  grain.visible = vis(MODE.grain);
  parts.push(grain);

  const dial = angleDial(node, {
    mode: "polar",
    angle: W.motionBlur.angle,
    distance: W.motionBlur.distance,
    maxDistance: 300,
    zeroLabel: "no blur",
  }, bump);
  dial.visible = vis(MODE.motionBlur);
  parts.push(dial);

  const colour = colourRow(node, W.colour.color, bump);
  colour.visible = vis(MODE.colour);
  parts.push(colour);

  const tol = tolerancePreview(node, W.colour);
  tol.visible = vis(MODE.colour);
  parts.push(tol);

  return mountParts(node, ST, parts);
}

app.registerExtension({
  name: "C2C.MaskToolkit",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const name = String(nodeData?.name || "");
    if (name !== NODE) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      try {
        attachMaskTools(this);
      } catch (_e) { /* never break the node */ }
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
      const r = onConfigure?.apply(this, args);
      this[ST]?.invalidate?.();
      return r;
    };

    // The gradient preview reports whether size_as is connected, so it has to
    // repaint when a wire lands or leaves.
    const onConnections = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (...args) {
      const r = onConnections?.apply(this, args);
      this[ST]?.invalidate?.();
      return r;
    };
  },
});
