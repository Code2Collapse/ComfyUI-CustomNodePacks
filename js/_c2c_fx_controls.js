// _c2c_fx_controls.js — controls shared by the layer-effect and mask-toolkit
// front-ends. Registers nothing; it is imported, not loaded for its own sake.
//
// Two of these exist because the default widget is genuinely unreadable:
//
//   colourRow  — seven nodes take their colour as a STRING holding "#FFBF30".
//                Nobody knows what #FFBF30 looks like.
//   angleDial  — several take a direction as two integers or as a lone degree
//                figure. Nobody thinks in x/y, and a number is not a direction.
//
// Plain ES module. Nothing here touches `window` or ComfyUI at import time, so
// it is safe in a headless run and safe to import from either family.

export const HEX_RE = /^#?[0-9a-fA-F]{6}$/;

export function css(el, s) { Object.assign(el.style, s); }

export function widgetsOf(node) {
  const map = {};
  for (const w of node.widgets || []) map[w.name] = w;
  return map;
}

export function normHex(v, fallback = "#000000") {
  const s = String(v ?? "").trim();
  if (!HEX_RE.test(s)) return fallback;
  return s.startsWith("#") ? s.toLowerCase() : "#" + s.toLowerCase();
}

export function hexToRgb(hex) {
  const h = normHex(hex);
  return [parseInt(h.slice(1, 3), 16),
          parseInt(h.slice(3, 5), 16),
          parseInt(h.slice(5, 7), 16)];
}

/** The checkerboard used wherever alpha or a dark colour has to stay legible. */
export function checkerCss(size = 8) {
  const h = size / 2;
  return {
    backgroundImage:
      `linear-gradient(45deg,#555 25%,transparent 25%,transparent 75%,#555 75%),` +
      `linear-gradient(45deg,#555 25%,#333 25%,#333 75%,#555 75%)`,
    backgroundSize: `${size}px ${size}px`,
    backgroundPosition: `0 0, ${h}px ${h}px`,
  };
}

// ── colour row ──────────────────────────────────────────────────────────────

/**
 * A swatch, the OS colour picker and an editable hex field, bound to one
 * STRING widget. The hex stays typeable because matching a value from a grade
 * or a brand guide means pasting it, not hunting for it in a picker.
 */
export function colourRow(node, name, onChange) {
  const row = document.createElement("div");
  css(row, { display: "flex", alignItems: "center", gap: "5px",
             padding: "1px 0", minWidth: "0" });

  const label = document.createElement("span");
  label.textContent = name.replace(/_/g, " ");
  css(label, { flex: "0 0 auto", opacity: ".75", minWidth: "62px" });

  const swatchWrap = document.createElement("span");
  css(swatchWrap, { flex: "0 0 auto", borderRadius: "3px", lineHeight: "0",
                    ...checkerCss() });

  const swatch = document.createElement("button");
  swatch.type = "button";
  swatch.title = `Pick ${name.replace(/_/g, " ")}`;
  css(swatch, { width: "22px", height: "14px", padding: "0", borderRadius: "3px",
                border: "1px solid rgba(255,255,255,0.28)", cursor: "pointer",
                background: "#000" });
  swatchWrap.append(swatch);

  const hex = document.createElement("input");
  hex.type = "text";
  hex.spellcheck = false;
  hex.title = "Hex value — paste one from anywhere";
  css(hex, { flex: "1 1 auto", minWidth: "0", width: "100%",
             background: "var(--comfy-input-bg,#222)", color: "inherit",
             border: "1px solid var(--border-color,#444)", borderRadius: "3px",
             padding: "0 4px", font: "inherit",
             fontVariantNumeric: "tabular-nums" });

  // Off-screen rather than styled: browsers give <input type=color> a look
  // that cannot be themed to match the node.
  const picker = document.createElement("input");
  picker.type = "color";
  css(picker, { position: "absolute", width: "0", height: "0",
                opacity: "0", pointerEvents: "none" });

  row.append(label, swatchWrap, hex, picker);

  const widget = () => widgetsOf(node)[name];

  const paint = () => {
    const w = widget();
    if (!w) return;
    const v = normHex(w.value);
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
    onChange?.();
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

  return { row, paint, height: 18 };
}

// ── angle dial ──────────────────────────────────────────────────────────────

export const DIAL_PX = 62;

/**
 * A direction-and-distance dial.
 *
 * `mode` is either:
 *   "xy"    — reads and writes two widgets holding pixel offsets
 *   "polar" — reads and writes an angle widget and a distance widget
 *
 * Both exist because both shapes exist in the nodes: a drop shadow is
 * specified as an offset, a motion blur as an angle and a length. The control
 * is the same in either case, because the user's question is the same one:
 * which way, and how far.
 */
export function angleDial(node, opts, onChange) {
  const { mode = "xy", maxDistance = 200 } = opts;

  const wrap = document.createElement("div");
  css(wrap, { display: "flex", alignItems: "center", gap: "8px", padding: "2px 0" });

  const cv = document.createElement("canvas");
  cv.width = DIAL_PX * 2;
  cv.height = DIAL_PX * 2;
  css(cv, { width: `${DIAL_PX}px`, height: `${DIAL_PX}px`, flex: "0 0 auto",
            cursor: "grab", borderRadius: "50%" });
  cv.title = "Drag to aim. Shift-drag holds the distance, Alt-drag holds the angle.";

  const readout = document.createElement("div");
  css(readout, { flex: "1 1 auto", minWidth: "0", lineHeight: "1.45",
                 whiteSpace: "pre-line", fontVariantNumeric: "tabular-nums",
                 opacity: ".85" });

  wrap.append(cv, readout);

  /** Current state as {deg, dist}, whatever the underlying widgets are. */
  const read = () => {
    const w = widgetsOf(node);
    if (mode === "polar") {
      return { deg: Number(w[opts.angle]?.value ?? 0),
               dist: Number(w[opts.distance]?.value ?? 0) };
    }
    const x = Number(w[opts.x]?.value ?? 0);
    const y = Number(w[opts.y]?.value ?? 0);
    const dist = Math.hypot(x, y);
    // Screen y grows downward; report the angle the way a compositor says it,
    // anticlockwise from east.
    const deg = dist === 0 ? 0 : ((-Math.atan2(y, x) * 180) / Math.PI + 360) % 360;
    return { deg, dist };
  };

  const write = (deg, dist) => {
    const w = widgetsOf(node);
    const set = (name, v) => {
      const wi = w[name];
      if (!wi) return;
      wi.value = v;
      wi.callback?.(v);
    };
    if (mode === "polar") {
      set(opts.angle, Math.round(((deg % 360) + 360) % 360));
      set(opts.distance, Math.round(dist));
    } else {
      const rad = (-deg * Math.PI) / 180;
      set(opts.x, Math.round(Math.cos(rad) * dist));
      set(opts.y, Math.round(Math.sin(rad) * dist));
    }
    node.setDirtyCanvas(true, true);
  };

  const paint = () => {
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const s = cv.width;
    const c = s / 2;
    ctx.clearRect(0, 0, s, s);

    ctx.fillStyle = "#1b1b1b";
    ctx.beginPath();
    ctx.arc(c, c, c - 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.16)";
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.strokeStyle = "rgba(255,255,255,0.10)";
    ctx.beginPath();
    ctx.moveTo(6, c); ctx.lineTo(s - 6, c);
    ctx.moveTo(c, 6); ctx.lineTo(c, s - 6);
    ctx.stroke();

    const { deg, dist } = read();
    // Distance compresses with a sqrt so a 25px and a 400px setting are both
    // readable on the same 62px control.
    const reach = dist === 0 ? 0 : Math.min(1, Math.sqrt(dist / maxDistance));
    const rad = (-deg * Math.PI) / 180;
    const hx = c + Math.cos(rad) * reach * (c - 10);
    const hy = c + Math.sin(rad) * reach * (c - 10);

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

    const lines = [`${deg.toFixed(0)}° · ${dist.toFixed(0)} px`];
    if (mode === "xy") {
      const w = widgetsOf(node);
      lines.push(`x ${w[opts.x]?.value ?? 0}  y ${w[opts.y]?.value ?? 0}`);
    }
    if (dist === 0) lines.push(opts.zeroLabel || "no offset");
    readout.textContent = lines.join("\n");
  };

  let drag = null;
  const move = (e) => {
    if (!drag) return;
    const r = cv.getBoundingClientRect();
    const dx = e.clientX - (r.left + r.width / 2);
    const dy = e.clientY - (r.top + r.height / 2);
    const reach = Math.min(1, Math.hypot(dx, dy) / (r.width / 2 - 5));
    const deg = e.altKey ? drag.deg0
      : ((-Math.atan2(dy, dx) * 180) / Math.PI + 360) % 360;
    const dist = e.shiftKey ? drag.dist0 : Math.round(reach * reach * maxDistance);
    write(deg, dist);
    paint();
    onChange?.();
  };
  cv.onpointerdown = (e) => {
    cv.setPointerCapture(e.pointerId);
    cv.style.cursor = "grabbing";
    const cur = read();
    drag = { deg0: cur.deg, dist0: cur.dist };
    move(e);
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

  return { wrap, paint, height: DIAL_PX + 4 };
}

// ── generic plumbing ────────────────────────────────────────────────────────

/**
 * Wire a set of parts into a node as one DOM widget, repainting on any widget
 * change, coalesced to one frame. Returns the state object, or null if there
 * was nothing to show.
 */
export function mountParts(node, stateKey, parts, extraHeight = 8) {
  if (!parts.length) return null;

  const root = document.createElement("div");
  css(root, { width: "100%", boxSizing: "border-box", padding: "3px 2px",
              display: "flex", flexDirection: "column", gap: "2px",
              font: "10px system-ui,sans-serif", color: "var(--input-text,#ddd)" });
  for (const p of parts) root.append(p.wrap || p.row);

  const st = { parts, raf: 0, dead: false };
  node[stateKey] = st;

  st.invalidate = () => {
    if (st.dead || st.raf) return;
    st.raf = requestAnimationFrame(() => {
      st.raf = 0;
      if (st.dead || !root.isConnected) return;
      for (const p of parts) {
        // One bad part must never blank the rest of the strip.
        try { p.paint(); } catch (_e) { /* keep going */ }
      }
    });
  };

  const total = parts.reduce((a, p) => a + (p.height || 18), extraHeight);
  const widget = node.addDOMWidget(stateKey, "div", root, { serialize: false });
  widget.computeSize = (width) => [width, total];

  // Repaint on ANY widget change: these controls read several widgets each, so
  // watching only "their own" would leave them stale.
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
