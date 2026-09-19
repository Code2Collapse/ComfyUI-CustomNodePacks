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

const HEX_RE = /^#?[0-9a-fA-F]{6}$/;

// ── helpers ─────────────────────────────────────────────────────────────────

function css(el, s) { Object.assign(el.style, s); }

function widgetsOf(node) {
  const map = {};
  for (const w of node.widgets || []) map[w.name] = w;
  return map;
}

function normHex(v, fallback = "#000000") {
  const s = String(v ?? "").trim();
  if (!HEX_RE.test(s)) return fallback;
  return s.startsWith("#") ? s.toLowerCase() : "#" + s.toLowerCase();
}

/** Which widgets on this node hold a colour. */
function colourWidgetNames(node) {
  return (node.widgets || [])
    .filter((w) => w.name === "color" || /_color$/.test(w.name))
    .map((w) => w.name);
}

// ── colour row ──────────────────────────────────────────────────────────────

function buildColourRow(node, name, invalidate) {
  const row = document.createElement("div");
  css(row, {
    display: "flex", alignItems: "center", gap: "5px",
    padding: "1px 0", minWidth: "0",
  });

  const label = document.createElement("span");
  label.textContent = name.replace(/_/g, " ");
  css(label, { flex: "0 0 auto", opacity: ".75", minWidth: "62px" });

  const swatch = document.createElement("button");
  swatch.type = "button";
  swatch.title = `Pick ${name.replace(/_/g, " ")}`;
  css(swatch, {
    flex: "0 0 auto", width: "22px", height: "14px", padding: "0",
    borderRadius: "3px", border: "1px solid rgba(255,255,255,0.28)",
    cursor: "pointer", background: "#000",
  });

  // A checkerboard behind the swatch, so a dark colour on a dark node is still
  // clearly a colour and not an empty hole.
  const swatchWrap = document.createElement("span");
  css(swatchWrap, {
    flex: "0 0 auto", borderRadius: "3px", padding: "0", lineHeight: "0",
    backgroundImage:
      "linear-gradient(45deg,#555 25%,transparent 25%,transparent 75%,#555 75%)," +
      "linear-gradient(45deg,#555 25%,#333 25%,#333 75%,#555 75%)",
    backgroundSize: "8px 8px",
    backgroundPosition: "0 0, 4px 4px",
  });
  swatchWrap.append(swatch);

  const hex = document.createElement("input");
  hex.type = "text";
  hex.spellcheck = false;
  hex.title = "Hex value — paste one from anywhere";
  css(hex, {
    flex: "1 1 auto", minWidth: "0", width: "100%",
    background: "var(--comfy-input-bg,#222)", color: "inherit",
    border: "1px solid var(--border-color,#444)", borderRadius: "3px",
    padding: "0 4px", font: "inherit", fontVariantNumeric: "tabular-nums",
  });

  // The native picker. Kept off-screen rather than styled, because browsers
  // give <input type=color> a look that cannot be themed to match the node.
  const picker = document.createElement("input");
  picker.type = "color";
  css(picker, { position: "absolute", width: "0", height: "0",
                opacity: "0", pointerEvents: "none" });

  row.append(label, swatchWrap, hex, picker);

  const widget = () => widgetsOf(node)[name];

  const paint = () => {
    const w = widget();
    if (!w) return;
    const v = normHex(w.value, "#000000");
    swatch.style.background = v;
    if (document.activeElement !== hex) hex.value = v;
    hex.style.borderColor = HEX_RE.test(String(w.value ?? "").trim())
      ? "var(--border-color,#444)" : "#c0564f";
  };

  const commit = (v) => {
    const w = widget();
    if (!w) return;
    w.value = v;
    w.callback?.(v);
    paint();
    invalidate();
    node.setDirtyCanvas(true, true);
  };

  swatch.onclick = () => { picker.value = normHex(widget()?.value); picker.click(); };
  picker.oninput = () => commit(picker.value);
  hex.oninput = () => {
    const raw = hex.value.trim();
    if (HEX_RE.test(raw)) commit(raw.startsWith("#") ? raw : "#" + raw);
    else hex.style.borderColor = "#c0564f";   // say it is wrong, do not fight it
  };
  hex.onblur = paint;

  return { row, paint };
}

// ── direction dial ──────────────────────────────────────────────────────────

const DIAL_PX = 62;

function buildDial(node, invalidate) {
  const wrap = document.createElement("div");
  css(wrap, { display: "flex", alignItems: "center", gap: "8px", padding: "2px 0" });

  const cv = document.createElement("canvas");
  cv.width = DIAL_PX * 2;
  cv.height = DIAL_PX * 2;
  css(cv, { width: `${DIAL_PX}px`, height: `${DIAL_PX}px`, flex: "0 0 auto",
            cursor: "grab", borderRadius: "50%" });
  cv.title = "Drag to aim the shadow. Shift-drag holds the distance, " +
             "Alt-drag holds the angle.";

  const readout = document.createElement("div");
  css(readout, { flex: "1 1 auto", minWidth: "0", lineHeight: "1.45",
                 fontVariantNumeric: "tabular-nums", opacity: ".85" });

  wrap.append(cv, readout);

  const xy = () => {
    const w = widgetsOf(node);
    return [Number(w.distance_x?.value ?? 0), Number(w.distance_y?.value ?? 0)];
  };

  const setXY = (x, y) => {
    const w = widgetsOf(node);
    if (w.distance_x) { w.distance_x.value = Math.round(x); w.distance_x.callback?.(w.distance_x.value); }
    if (w.distance_y) { w.distance_y.value = Math.round(y); w.distance_y.callback?.(w.distance_y.value); }
    node.setDirtyCanvas(true, true);
  };

  const paint = () => {
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const s = cv.width;
    const c = s / 2;
    ctx.clearRect(0, 0, s, s);

    // dial face
    ctx.fillStyle = "#1b1b1b";
    ctx.beginPath();
    ctx.arc(c, c, c - 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.16)";
    ctx.lineWidth = 2;
    ctx.stroke();

    // cross hairs
    ctx.strokeStyle = "rgba(255,255,255,0.10)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(6, c); ctx.lineTo(s - 6, c);
    ctx.moveTo(c, 6); ctx.lineTo(c, s - 6);
    ctx.stroke();

    const [x, y] = xy();
    const dist = Math.hypot(x, y);
    // The dial shows direction at full radius and distance as the handle's
    // reach, compressed with a sqrt so a 25px and a 400px shadow are both
    // readable on the same 62px control.
    const reach = dist === 0 ? 0 : Math.min(1, Math.sqrt(dist / 200));
    const hx = c + (dist === 0 ? 0 : (x / dist) * reach * (c - 10));
    const hy = c + (dist === 0 ? 0 : (y / dist) * reach * (c - 10));

    ctx.strokeStyle = "#e0a24a";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(c, c);
    ctx.lineTo(hx, hy);
    ctx.stroke();

    ctx.fillStyle = "#e0a24a";
    ctx.beginPath();
    ctx.arc(hx, hy, 6, 0, Math.PI * 2);
    ctx.fill();

    // Screen y grows downward; report the angle the way a compositor says it,
    // anticlockwise from east.
    const deg = dist === 0 ? 0 : ((-Math.atan2(y, x) * 180) / Math.PI + 360) % 360;
    readout.textContent =
      `${deg.toFixed(0)}° · ${dist.toFixed(0)} px\n` +
      `x ${x}  y ${y}` +
      (dist === 0 ? "\ncentred — no offset" : "");
    readout.style.whiteSpace = "pre-line";
  };

  let drag = null;
  cv.onpointerdown = (e) => {
    cv.setPointerCapture(e.pointerId);
    cv.style.cursor = "grabbing";
    const [x0, y0] = xy();
    drag = { d0: Math.hypot(x0, y0), a0: Math.atan2(y0, x0) };
    move(e);
  };
  const move = (e) => {
    if (!drag) return;
    const r = cv.getBoundingClientRect();
    const dx = e.clientX - (r.left + r.width / 2);
    const dy = e.clientY - (r.top + r.height / 2);
    const reach = Math.min(1, Math.hypot(dx, dy) / (r.width / 2 - 5));
    const angle = e.altKey ? drag.a0 : Math.atan2(dy, dx);
    const dist = e.shiftKey ? drag.d0 : Math.round(reach * reach * 200);
    setXY(Math.cos(angle) * dist, Math.sin(angle) * dist);
    paint();
    invalidate();
  };
  cv.onpointermove = (e) => { if (drag) move(e); };
  const end = (e) => {
    if (!drag) return;
    drag = null;
    cv.style.cursor = "grab";
    try { cv.releasePointerCapture(e.pointerId); } catch (_e) { /* already gone */ }
  };
  cv.onpointerup = end;
  cv.onpointercancel = end;

  return { wrap, paint };
}

// ── gradient ramp ───────────────────────────────────────────────────────────

const RAMP_H = 26;

function buildRamp(node, cfg, invalidate) {
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

  return { wrap, paint };
}

// ── assembly ────────────────────────────────────────────────────────────────

function attach(node, nodeName) {
  if (node[ST]) return node[ST];

  const root = document.createElement("div");
  css(root, {
    width: "100%", boxSizing: "border-box", padding: "3px 2px",
    display: "flex", flexDirection: "column", gap: "2px",
    font: "10px system-ui,sans-serif", color: "var(--input-text,#ddd)",
  });

  const st = { parts: [], raf: 0, dead: false };
  node[ST] = st;

  st.invalidate = () => {
    if (st.dead || st.raf) return;
    st.raf = requestAnimationFrame(() => {
      st.raf = 0;
      if (st.dead || !root.isConnected) return;
      for (const p of st.parts) { try { p.paint(); } catch (_e) { /* one bad part
        must not blank the rest */ } }
    });
  };

  if (DIAL_NODES.has(nodeName)) {
    const dial = buildDial(node, st.invalidate);
    root.append(dial.wrap);
    st.parts.push(dial);
  }

  const rampCfg = RAMP_NODES[nodeName];
  if (rampCfg) {
    const ramp = buildRamp(node, rampCfg, st.invalidate);
    root.append(ramp.wrap);
    st.parts.push(ramp);
  }

  for (const name of colourWidgetNames(node)) {
    const row = buildColourRow(node, name, st.invalidate);
    root.append(row.row);
    st.parts.push(row);
  }

  if (!st.parts.length) { delete node[ST]; return null; }

  const rows = st.parts.length;
  const widget = node.addDOMWidget("c2c_layer_fx", "div", root, { serialize: false });
  widget.computeSize = (width) => {
    let h = 8;
    if (DIAL_NODES.has(nodeName)) h += DIAL_PX + 4;
    if (rampCfg) h += RAMP_H + 16;
    h += colourWidgetNames(node).length * 18;
    return [width, h];
  };
  void rows;

  // Repaint whenever any widget moves - the ramp and the dial both read several
  // widgets, so watching only "their own" would leave them stale.
  for (const w of node.widgets || []) {
    const prev = w.callback;
    w.callback = function (...args) {
      const r = prev?.apply(this, args);
      st.invalidate();
      return r;
    };
  }

  if (typeof ResizeObserver !== "undefined") {
    st.ro = new ResizeObserver(() => st.invalidate());
    st.ro.observe(root);
  }

  const onRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    st.dead = true;
    if (st.raf) cancelAnimationFrame(st.raf);
    st.ro?.disconnect();
    return onRemoved?.apply(this, args);
  };

  st.invalidate();
  return st;
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
