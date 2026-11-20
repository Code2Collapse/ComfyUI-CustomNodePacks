// _vault_ui.js - the C2C Vault design system (V1, owner 2026-11-20: "a polished, modern, professional interface",
// docs/h2vault/V1_cnp_vault_ux_spec.md). Dialog shell, fields, choice cards, notices, copy rows, the seal mark, the
// selection capsule and the vault node's status card.
//
// Rules: no import of the ComfyUI frontend (a wrong relative depth 404s the pack's whole front end); side-effect free
// at import (tests import it under node); never innerHTML - elements and text nodes only; tokens only (`--cu-*` from
// c2c_ui/theme.js, which read the live NIGHTWOLF palette), so a palette switch reaches the vault too.
//
// License: Apache-2.0
import { ensureStyles } from "./c2c_ui/theme.js";

// ── pure helpers (unit-tested) ─────────────────────────────────────────────────────────────────────────────────
/** Password strength for the meter. Under 8 characters is never accepted (the server rule). */
export function passwordStrength(pw) {
  const s = String(pw ?? "");
  if (!s) return { score: 0, label: "", ok: false };
  if (s.length < 8) return { score: 0, label: "Too short", ok: false };
  const classes = [/[a-z]/, /[A-Z]/, /[0-9]/, /[^a-zA-Z0-9]/].filter((re) => re.test(s)).length;
  const score = Math.min(4, 1 + (s.length >= 12 ? 1 : 0) + (s.length >= 16 ? 1 : 0) + (classes >= 3 ? 1 : 0));
  return { score, label: ["", "Fair", "Good", "Strong", "Very strong"][score], ok: true };
}

/** Seconds until an authenticator (TOTP) code changes, by this machine's clock. */
export function secondsLeft(nowMs, period = 30) {
  return period - (Math.floor(nowMs / 1000) % period);
}

// ── styles ─────────────────────────────────────────────────────────────────────────────────────────────────────
const STYLE_ID = "c2c-vault-ui-v1";
const CSS = `
.c2c-vault { --cv-r-dialog: 10px; --cv-r-card: 8px; --cv-r-ctl: 6px; font-family: inherit; color: var(--cu-ink);
  font-size: 13px; line-height: 1.45; }
.c2c-vault *, .c2c-vault *::before, .c2c-vault *::after { box-sizing: border-box; }
.c2c-vault-backdrop { position: fixed; inset: 0; z-index: 10000; display: flex; align-items: center;
  justify-content: center; padding: 16px; background: rgba(4, 4, 16, 0.62); }
.c2c-vault-panel { width: min(var(--cv-w, 480px), 94vw); max-height: 90vh; display: flex; flex-direction: column;
  background: var(--cu-panel); border: 1px solid var(--cu-edge-strong); border-radius: var(--cv-r-dialog);
  box-shadow: 0 24px 64px rgba(0, 0, 0, 0.55); outline: none; animation: c2c-vault-in 120ms ease-out; }
@keyframes c2c-vault-in { from { opacity: 0; transform: scale(0.98); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) { .c2c-vault-panel { animation: none; } }
.c2c-vault-head { display: flex; gap: 12px; align-items: flex-start; padding: 16px 16px 12px 20px;
  border-bottom: 1px solid var(--cu-edge); }
.c2c-vault-head__text { flex: 1; min-width: 0; padding-top: 2px; }
.c2c-vault-title { margin: 0; font-size: 15px; font-weight: 600; line-height: 1.3; color: var(--cu-ink);
  overflow-wrap: anywhere; }
.c2c-vault-sub { margin: 4px 0 0; font-size: 12.5px; color: var(--cu-ink-soft); }
.c2c-vault-x { flex: 0 0 auto; width: 32px; height: 32px; border-radius: var(--cv-r-ctl); border: 1px solid transparent;
  background: transparent; color: var(--cu-ink-soft); cursor: pointer; display: grid; place-items: center; }
.c2c-vault-x:hover:not(:disabled) { background: var(--cu-raised); color: var(--cu-ink); }
.c2c-vault-body { padding: 16px 20px; overflow: auto; display: flex; flex-direction: column; gap: 16px; }
.c2c-vault-slot:empty { display: none; }
.c2c-vault-slot { padding: 0 20px 12px; }
.c2c-vault-foot { display: flex; gap: 12px; align-items: center; padding: 12px 16px 16px 20px;
  border-top: 1px solid var(--cu-edge); }
.c2c-vault-status { flex: 1; min-width: 0; font-size: 12px; color: var(--cu-ink-soft); }
.c2c-vault-actions { display: flex; gap: 8px; flex: 0 0 auto; }
.c2c-vault :focus { outline: none; }
.c2c-vault :focus-visible { outline: 2px solid var(--cu-accent); outline-offset: 2px; }

.c2c-vault-btn { height: 32px; padding: 0 14px; border-radius: var(--cv-r-ctl); font: inherit; font-size: 12.5px;
  font-weight: 600; cursor: pointer; display: inline-flex; align-items: center; gap: 8px; white-space: nowrap;
  border: 1px solid var(--cu-edge-strong); background: transparent; color: var(--cu-ink); }
.c2c-vault-btn:hover:not(:disabled) { border-color: var(--cu-ink-dim); background: var(--cu-raised); }
.c2c-vault-btn--primary { background: var(--cu-accent); border-color: var(--cu-accent); color: var(--cu-on-accent); }
.c2c-vault-btn--primary:hover:not(:disabled) { background: var(--cu-accent-hover); border-color: var(--cu-accent-hover); }
.c2c-vault-btn--danger { color: var(--cu-danger); border-color: color-mix(in srgb, var(--cu-danger) 55%, transparent); }
.c2c-vault-btn--small { height: 26px; padding: 0 10px; font-size: 11.5px; }
.c2c-vault-btn--text { border-color: transparent; padding: 0 8px; color: var(--cu-ink-soft); font-weight: 500; }
.c2c-vault-btn--text:hover:not(:disabled) { color: var(--cu-ink); background: var(--cu-raised); border-color: transparent; }
.c2c-vault-btn:disabled { opacity: 0.55; cursor: default; }
.c2c-vault-btn--primary:disabled { opacity: 0.8; }
.c2c-vault-spin { width: 12px; height: 12px; border-radius: 50%; border: 2px solid currentColor;
  border-right-color: transparent; animation: c2c-vault-spin 0.7s linear infinite; }
@keyframes c2c-vault-spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) { .c2c-vault-spin { animation-duration: 2.4s; } }

.c2c-vault-field { display: flex; flex-direction: column; gap: 6px; }
.c2c-vault-field > div:empty { display: none; }
.c2c-vault-label { font-size: 12px; font-weight: 600; color: var(--cu-ink-soft); }
.c2c-vault-hint { font-size: 12px; color: var(--cu-ink-dim); }
.c2c-vault-input, .c2c-vault-textarea { width: 100%; height: 34px; padding: 0 10px; border-radius: var(--cv-r-ctl);
  border: 1px solid var(--cu-edge-strong); background: var(--cu-sunken); color: var(--cu-ink); font: inherit;
  font-size: 13px; }
.c2c-vault-textarea { height: auto; min-height: 64px; padding: 8px 10px; resize: vertical; font-family: ui-monospace,
  "Cascadia Mono", Consolas, monospace; font-size: 12px; }
.c2c-vault-input::placeholder, .c2c-vault-textarea::placeholder { color: var(--cu-ink-dim); }
.c2c-vault-input:hover:not(:disabled), .c2c-vault-textarea:hover:not(:disabled) { border-color: var(--cu-ink-dim); }
.c2c-vault-input:focus-visible, .c2c-vault-textarea:focus-visible { outline: none; border-color: var(--cu-accent);
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--cu-accent) 35%, transparent); }
.c2c-vault-input:disabled, .c2c-vault-textarea:disabled { opacity: 0.6; }
.c2c-vault-inputwrap { position: relative; display: flex; }
.c2c-vault-inputwrap .c2c-vault-input { padding-right: 64px; }
.c2c-vault-inputwrap .c2c-vault-btn { position: absolute; right: 3px; top: 3px; height: 28px; }
.c2c-vault-meter { display: flex; align-items: center; gap: 8px; min-height: 16px; flex: 1; min-width: 0; }
.c2c-vault-meter__bars { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; width: 120px; }
.c2c-vault-meter__bars span { height: 4px; border-radius: 2px; background: var(--cu-edge); }
.c2c-vault-meter__text { font-size: 11.5px; color: var(--cu-ink-soft); white-space: nowrap; }
.c2c-vault-row { display: flex; gap: 8px; align-items: center; }
.c2c-vault-grid2 { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; }
@media (max-width: 560px) { .c2c-vault-grid2 { grid-template-columns: minmax(0, 1fr); } }
.c2c-vault-cols { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.1fr); gap: 24px; align-items: start; }
.c2c-vault-col { display: flex; flex-direction: column; gap: 16px; min-width: 0; }
@media (max-width: 860px) { .c2c-vault-cols { grid-template-columns: minmax(0, 1fr); } }

.c2c-vault-code { height: 44px; text-align: center; font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
  font-size: 18px; letter-spacing: 0.3em; }
.c2c-vault-code::placeholder { letter-spacing: normal; font-family: inherit; font-size: 13px; }
.c2c-vault-clock { display: flex; align-items: center; gap: 8px; font-size: 11.5px; color: var(--cu-ink-dim); }
.c2c-vault-clock__bar { flex: 1; height: 3px; border-radius: 2px; background: var(--cu-edge); overflow: hidden; }
.c2c-vault-clock__bar span { display: block; height: 100%; background: var(--cu-accent); transition: width 1s linear; }
@media (prefers-reduced-motion: reduce) { .c2c-vault-clock__bar span { transition: none; } }

.c2c-vault-choices { display: grid; gap: 8px; }
.c2c-vault-choice { display: grid; grid-template-columns: 16px 1fr; gap: 4px 10px; padding: 10px 12px;
  border-radius: var(--cv-r-card); border: 1px solid var(--cu-edge-strong); background: var(--cu-sunken);
  cursor: pointer; text-align: left; }
.c2c-vault-choice:hover { border-color: var(--cu-ink-dim); }
.c2c-vault-choice[aria-checked="true"] { border-color: var(--cu-accent);
  background: color-mix(in srgb, var(--cu-accent) 10%, var(--cu-sunken)); }
.c2c-vault-choice__dot { grid-row: span 2; width: 16px; height: 16px; margin-top: 1px; border-radius: 50%;
  border: 1.5px solid var(--cu-ink-dim); display: grid; place-items: center; }
.c2c-vault-choice[aria-checked="true"] .c2c-vault-choice__dot { border-color: var(--cu-accent); }
.c2c-vault-choice[aria-checked="true"] .c2c-vault-choice__dot::after { content: ""; width: 8px; height: 8px;
  border-radius: 50%; background: var(--cu-accent); }
.c2c-vault-choice__title { display: flex; gap: 8px; align-items: center; font-weight: 600; font-size: 13px; }
.c2c-vault-choice__sum { font-size: 12px; color: var(--cu-ink-soft); }
.c2c-vault-tag { font-size: 11px; font-weight: 600; padding: 1px 8px; border-radius: 999px;
  border: 1px solid var(--cu-edge-strong); color: var(--cu-ink-soft); }
.c2c-vault-tag--ok { color: var(--cu-ok); border-color: color-mix(in srgb, var(--cu-ok) 45%, transparent); }
.c2c-vault-tag--warn { color: var(--cu-warn); border-color: color-mix(in srgb, var(--cu-warn) 45%, transparent); }
.c2c-vault-tag--accent { color: var(--cu-accent); border-color: color-mix(in srgb, var(--cu-accent) 45%, transparent); }

.c2c-vault-facts { display: grid; grid-template-columns: max-content 1fr; gap: 6px 16px; margin: 0; padding: 12px;
  border-radius: var(--cv-r-card); background: color-mix(in srgb, var(--cu-sunken) 60%, transparent);
  border: 1px solid var(--cu-edge); }
.c2c-vault-facts dt { font-size: 12px; color: var(--cu-ink-dim); }
.c2c-vault-facts dd { margin: 0; font-size: 12.5px; color: var(--cu-ink); }

.c2c-vault-notice { display: flex; gap: 10px; padding: 10px 12px; border-radius: var(--cv-r-card); font-size: 12.5px;
  border: 1px solid var(--cu-edge); border-left: 3px solid var(--cu-ink-dim); background: var(--cu-sunken);
  color: var(--cu-ink); white-space: pre-wrap; overflow-wrap: anywhere; }
.c2c-vault-notice--info { border-left-color: var(--cu-accent); }
.c2c-vault-notice--ok { border-left-color: var(--cu-ok); background: color-mix(in srgb, var(--cu-ok) 8%, var(--cu-sunken)); }
.c2c-vault-notice--warn { border-left-color: var(--cu-warn); background: color-mix(in srgb, var(--cu-warn) 8%, var(--cu-sunken)); }
.c2c-vault-notice--danger { border-left-color: var(--cu-danger); background: color-mix(in srgb, var(--cu-danger) 9%, var(--cu-sunken)); }

.c2c-vault-copy { display: flex; gap: 8px; align-items: stretch; }
.c2c-vault-copy__val { flex: 1; min-width: 0; padding: 8px 10px; border-radius: var(--cv-r-ctl); background: var(--cu-sunken);
  border: 1px solid var(--cu-edge); font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; font-size: 12px;
  word-break: break-all; user-select: all; color: var(--cu-ink); }
.c2c-vault-copy .c2c-vault-btn { align-self: center; }

.c2c-vault-capsule { display: flex; flex-direction: column; gap: 8px; padding: 12px; border-radius: var(--cv-r-card);
  border: 1px dashed var(--cu-edge-strong); background: color-mix(in srgb, var(--cu-sunken) 70%, transparent); }
.c2c-vault-capsule__flow { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.c2c-vault-chips { display: flex; gap: 6px; flex-wrap: wrap; flex: 1; min-width: 0; }
.c2c-vault-chip { max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12px;
  padding: 3px 10px; border-radius: 999px; background: var(--cu-raised); border: 1px solid var(--cu-edge-strong);
  color: var(--cu-ink); }
.c2c-vault-chip--more { color: var(--cu-ink-soft); background: transparent; }
.c2c-vault-arrow { color: var(--cu-ink-dim); flex: 0 0 auto; }
.c2c-vault-capsule__meta { font-size: 12px; color: var(--cu-ink-soft); }

.c2c-vault-seal { flex: 0 0 auto; display: grid; place-items: center; border-radius: 8px; color: var(--cu-accent);
  background: color-mix(in srgb, currentColor 12%, var(--cu-sunken));
  box-shadow: inset 0 0 0 1px color-mix(in srgb, currentColor 45%, transparent); }
.c2c-vault-seal--ok { color: var(--cu-ok); }
.c2c-vault-seal--warn { color: var(--cu-warn); }
.c2c-vault-seal--danger { color: var(--cu-danger); }

.c2c-vault-disc { border: 1px solid var(--cu-edge); border-radius: var(--cv-r-card); }
.c2c-vault-disc > summary { list-style: none; cursor: pointer; padding: 8px 12px; font-size: 12.5px; font-weight: 600;
  color: var(--cu-ink-soft); display: flex; gap: 8px; align-items: center; border-radius: var(--cv-r-card); }
.c2c-vault-disc > summary::-webkit-details-marker { display: none; }
.c2c-vault-disc > summary::before { content: ""; width: 6px; height: 6px; border-right: 1.5px solid currentColor;
  border-bottom: 1.5px solid currentColor; transform: rotate(-45deg); transition: transform 120ms; }
.c2c-vault-disc[open] > summary::before { transform: rotate(45deg); }
.c2c-vault-disc__body { padding: 4px 12px 12px; display: flex; flex-direction: column; gap: 8px; }

.c2c-vault-switch { display: flex; align-items: center; gap: 10px; background: none; border: 0; padding: 0;
  color: var(--cu-ink); font: inherit; font-size: 13px; font-weight: 600; cursor: pointer; }
.c2c-vault-switch__track { width: 30px; height: 18px; border-radius: 999px; background: var(--cu-edge-strong);
  position: relative; transition: background 120ms; }
.c2c-vault-switch__track::after { content: ""; position: absolute; top: 2px; left: 2px; width: 14px; height: 14px;
  border-radius: 50%; background: var(--cu-ink); transition: transform 120ms; }
.c2c-vault-switch[aria-checked="true"] .c2c-vault-switch__track { background: var(--cu-accent); }
.c2c-vault-switch[aria-checked="true"] .c2c-vault-switch__track::after { transform: translateX(12px);
  background: var(--cu-on-accent); }

.c2c-vault-step { display: grid; grid-template-columns: 24px 1fr; gap: 4px 12px; }
.c2c-vault-step__n { width: 24px; height: 24px; border-radius: 50%; display: grid; place-items: center; font-size: 12px;
  font-weight: 700; color: var(--cu-accent); box-shadow: inset 0 0 0 1.5px var(--cu-accent); }
.c2c-vault-step__title { margin: 0; font-size: 13px; font-weight: 600; align-self: center; }
.c2c-vault-step__body { grid-column: 2; display: flex; flex-direction: column; gap: 8px; }

.c2c-vault-badge { font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 999px; white-space: nowrap;
  color: var(--cu-ok); background: color-mix(in srgb, var(--cu-ok) 12%, transparent); }
.c2c-vault-keyrow { display: grid; grid-template-columns: 160px 1fr auto; gap: 8px; align-items: center; }
.c2c-vault-keyrow .c2c-vault-input { height: 30px; font-size: 12.5px; }
.c2c-vault-keycode { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12px;
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; color: var(--cu-ink-soft); }
.c2c-vault-list { display: flex; flex-direction: column; gap: 6px; }
.c2c-vault-block { display: flex; flex-direction: column; gap: 8px; }
.c2c-vault-block__title { margin: 0; font-size: 13px; font-weight: 600; }
.c2c-vault-block__hint { margin: -4px 0 0; font-size: 12px; color: var(--cu-ink-dim); }
.c2c-vault-check { display: flex; gap: 8px; align-items: center; font-size: 12.5px; cursor: pointer; }
.c2c-vault-check input { width: 16px; height: 16px; accent-color: var(--cu-accent); margin: 0; }
.c2c-vault-qr { flex: 0 0 auto; width: 184px; height: 184px; padding: 8px; border-radius: var(--cv-r-card);
  background: #ffffff; }
.c2c-vault-qr img { display: block; width: 168px; height: 168px; image-rendering: pixelated; }
.c2c-vault-codes { display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px 16px; padding: 10px 12px;
  border-radius: var(--cv-r-ctl); background: var(--cu-sunken); border: 1px solid var(--cu-edge);
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; font-size: 12.5px; user-select: all; }
.c2c-vault-dim { font-size: 12px; color: var(--cu-ink-dim); margin: 0; }

.c2c-vault-card { display: flex; gap: 10px; align-items: center; width: 100%; height: 100%; padding: 8px 10px;
  border-radius: var(--cv-r-card); background: color-mix(in srgb, var(--cu-sunken) 85%, transparent);
  border: 1px solid var(--cu-edge); overflow: hidden; }
.c2c-vault-card__text { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 1px; }
.c2c-vault-card__title { font-size: 12.5px; font-weight: 600; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; color: var(--cu-ink); }
.c2c-vault-card__title--warn { color: var(--cu-warn); }
.c2c-vault-card__detail { font-size: 11.5px; color: var(--cu-ink-soft); white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }
.c2c-vault-card__meta { font-size: 11px; color: var(--cu-ink-dim); white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }
.c2c-vault-card__meta--ok { color: var(--cu-ok); }
`;

export function ensureVaultStyles() {
  if (typeof document === "undefined") return;
  ensureStyles();
  if (document.getElementById(STYLE_ID)) return;
  const s = document.createElement("style");
  s.id = STYLE_ID;
  s.textContent = CSS;
  document.head.appendChild(s);
}

// ── tiny DOM helpers ───────────────────────────────────────────────────────────────────────────────────────────
let UID = 0;
const uid = (p) => `c2c-vault-${p}-${++UID}`;

/** el("div", {class, text, attrs..., on: {click}}, children) - text only, never HTML. */
export function el(tag, props = {}, kids = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k === "style") Object.assign(e.style, v);
    else if (k === "on") for (const [ev, fn] of Object.entries(v)) e.addEventListener(ev, fn);
    else if (k in e && typeof e[k] !== "function" && !k.startsWith("aria") && k !== "role") {
      try { e[k] = v; } catch (_) { e.setAttribute(k, v); }
    } else e.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of [].concat(kids)) if (c != null && c !== false) e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return e;
}

const SVG = "http://www.w3.org/2000/svg";
function svg(tag, attrs = {}, kids = []) {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  for (const c of kids) e.append(c);
  return e;
}

/** The seal: a padlock in a ringed tile - the Vault's one recurring mark. */
export function sealGlyph({ size = 36, tone = "accent" } = {}) {
  ensureVaultStyles();
  const g = Math.round(size * 0.56);
  const icon = svg("svg", { width: g, height: g, viewBox: "0 0 24 24", fill: "none", "aria-hidden": "true",
    focusable: "false" }, [
    svg("path", { d: "M7.5 10.5V8a4.5 4.5 0 0 1 9 0v2.5", stroke: "currentColor", "stroke-width": "1.8",
      "stroke-linecap": "round" }),
    svg("rect", { x: "4.5", y: "10.5", width: "15", height: "10", rx: "2.2", fill: "currentColor", "fill-opacity": "0.18",
      stroke: "currentColor", "stroke-width": "1.8" }),
    svg("circle", { cx: "12", cy: "14.6", r: "1.5", fill: "currentColor" }),
    svg("path", { d: "M12 15.8v2", stroke: "currentColor", "stroke-width": "1.8", "stroke-linecap": "round" }),
  ]);
  const tile = el("span", { class: `c2c-vault-seal${tone && tone !== "accent" ? ` c2c-vault-seal--${tone}` : ""}`,
    "aria-hidden": "true", style: { width: `${size}px`, height: `${size}px`, borderRadius: `${Math.round(size / 4.5)}px` } },
  [icon]);
  return tile;
}

function closeIcon() {
  return svg("svg", { width: 14, height: 14, viewBox: "0 0 14 14", "aria-hidden": "true", focusable: "false" }, [
    svg("path", { d: "M3 3l8 8M11 3l-8 8", stroke: "currentColor", "stroke-width": "1.6", "stroke-linecap": "round" })]);
}

function arrowIcon() {
  return svg("svg", { width: 28, height: 14, viewBox: "0 0 28 14", class: "c2c-vault-arrow", "aria-hidden": "true",
    focusable: "false" }, [
    svg("path", { d: "M2 7h22M19 2.5L24 7l-5 4.5", stroke: "currentColor", "stroke-width": "1.6", fill: "none",
      "stroke-linecap": "round", "stroke-linejoin": "round" })]);
}

/** A button. kind: "primary" | "secondary" (default) | "danger" | "text"; small: compact height. */
export function vbutton(label, { kind = "secondary", small = false, onClick, ariaLabel, type = "button" } = {}) {
  ensureVaultStyles();
  const b = el("button", { type, class: `c2c-vault-btn${kind !== "secondary" ? ` c2c-vault-btn--${kind}` : ""}` +
    (small ? " c2c-vault-btn--small" : ""), "aria-label": ariaLabel || null });
  b.append(el("span", { class: "c2c-vault-btn__label", text: label }));
  if (onClick) b.addEventListener("click", onClick);
  return b;
}

function setButtonBusy(b, on, label) {
  if (!b) return;
  if (on) {
    if (b.__vaultLabel == null) b.__vaultLabel = b.querySelector(".c2c-vault-btn__label")?.textContent ?? b.textContent;
    b.replaceChildren(el("span", { class: "c2c-vault-spin", "aria-hidden": "true" }),
      el("span", { class: "c2c-vault-btn__label", text: label || b.__vaultLabel }));
  } else if (b.__vaultLabel != null) {
    b.replaceChildren(el("span", { class: "c2c-vault-btn__label", text: b.__vaultLabel }));
    b.__vaultLabel = null;
  }
}

// ── the dialog shell ───────────────────────────────────────────────────────────────────────────────────────────
const FOCUSABLE = "button:not([disabled]), input:not([disabled]):not([type=hidden]), textarea:not([disabled]), " +
  "select:not([disabled]), summary, [tabindex]:not([tabindex='-1'])";

/**
 * A modal dialog. Returns a controller; `result` resolves with whatever close() gets (Esc / the close button: null).
 * Focus is trapped while open and returned on close; key presses never reach the canvas behind it.
 */
export function vaultDialog({ title, subtitle = "", width = 480, tone = "accent" } = {}) {
  ensureVaultStyles();
  const tid = uid("title"), sid = uid("sub");
  const prevFocus = document.activeElement;
  let busy = false, done = false, submit = null, resolveFn;
  const cleanups = [];
  const result = new Promise((r) => { resolveFn = r; });
  const xBtn = el("button", { type: "button", class: "c2c-vault-x", "aria-label": "Close" }, [closeIcon()]);
  const titleEl = el("h2", { class: "c2c-vault-title", id: tid, text: title });
  const head = el("div", { class: "c2c-vault-head" }, [sealGlyph({ size: 36, tone }),
    el("div", { class: "c2c-vault-head__text" }, [titleEl, subtitle ? el("p", { class: "c2c-vault-sub", id: sid,
      text: subtitle }) : null]), xBtn]);
  const body = el("div", { class: "c2c-vault-body" });
  const slot = el("div", { class: "c2c-vault-slot" });
  const statusEl = el("div", { class: "c2c-vault-status", role: "status", "aria-live": "polite" });
  const actions = el("div", { class: "c2c-vault-actions" });
  const footer = el("div", { class: "c2c-vault-foot" }, [statusEl, actions]);
  const panel = el("div", { class: "c2c-vault-panel", role: "dialog", "aria-modal": "true", "aria-labelledby": tid,
    "aria-describedby": subtitle ? sid : null, tabindex: "-1", style: { "--cv-w": `${width}px` } },
  [head, body, slot, footer]);
  panel.style.setProperty("--cv-w", `${width}px`);
  const back = el("div", { class: "c2c-ui c2c-vault c2c-vault-backdrop" }, [panel]);
  let primaryBtn = null;

  const close = (value = null) => {
    if (done) return;
    done = true;
    for (const fn of cleanups.splice(0)) { try { fn(); } catch (_) { /* cleanup only */ } }
    for (const i of panel.querySelectorAll("input[type=password], input.c2c-vault-secret")) i.value = "";
    back.remove();
    try { if (prevFocus && typeof prevFocus.focus === "function" && document.contains(prevFocus)) prevFocus.focus(); }
    catch (_) { /* focus is a nicety */ }
    resolveFn(value);
  };
  xBtn.addEventListener("click", () => { if (!busy) close(null); });
  panel.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      e.preventDefault();
      if (!busy) close(null);
    } else if (e.key === "Tab") {
      const f = [...panel.querySelectorAll(FOCUSABLE)].filter((x) => x.offsetParent !== null || x === document.activeElement);
      if (f.length) {
        const first = f[0], last = f[f.length - 1];
        if (e.shiftKey && (document.activeElement === first || !panel.contains(document.activeElement))) {
          e.preventDefault(); last.focus();
        } else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      }
    } else if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      const t = e.target;
      if (t?.tagName === "INPUT" && t.type !== "checkbox" && submit && !busy) { e.preventDefault(); submit(); }
    }
    e.stopPropagation();        // canvas shortcuts (h, v, Delete, Ctrl+Z ...) never fire while a dialog is open
  });
  for (const ev of ["keyup", "keypress"]) panel.addEventListener(ev, (e) => e.stopPropagation());
  document.body.append(back);

  const api = {
    root: back, panel, body, footer, result, close,
    get busy() { return busy; },
    addAction(label, { primary = false, danger = false, onClick } = {}) {
      const b = vbutton(label, { kind: primary ? "primary" : danger ? "danger" : "secondary", onClick });
      actions.append(b);
      if (primary) primaryBtn = b;
      return b;
    },
    setBusy(on, label) {
      busy = !!on;
      panel.setAttribute("aria-busy", busy ? "true" : "false");
      for (const c of panel.querySelectorAll("input, button, textarea, select")) {
        if (busy) { if (!c.disabled) { c.disabled = true; c.__vaultBusy = true; } }
        else if (c.__vaultBusy) { c.disabled = false; c.__vaultBusy = false; }
      }
      setButtonBusy(primaryBtn, busy, label);
    },
    status(text) { statusEl.textContent = text || ""; },
    notice(tone, text) {
      slot.replaceChildren();
      if (tone && text) slot.append(noticeEl(tone, text));
    },
    onSubmit(fn) { submit = fn; },
    onCleanup(fn) { cleanups.push(fn); },
    focusFirst() {
      const f = panel.querySelector("input:not([disabled]):not([type=checkbox]), textarea:not([disabled])")
        ?? panel.querySelector(FOCUSABLE);
      (f ?? panel).focus();
    },
  };
  setTimeout(() => { if (!done) api.focusFirst(); }, 0);
  return api;
}

// ── fields ─────────────────────────────────────────────────────────────────────────────────────────────────────
/** A labelled row around any control. */
export function field({ label, hint, control, id }) {
  ensureVaultStyles();
  const cid = id || control?.id || uid("f");
  const target = control?.matches?.("input, textarea, select") ? control : control?.querySelector?.("input, textarea, select");
  if (target && !target.id) target.id = cid;
  const lab = el("label", { class: "c2c-vault-label", text: label, for: target?.id || cid });
  const kids = [lab, control];
  if (hint) {
    const h = el("div", { class: "c2c-vault-hint", id: uid("h"), text: hint });
    if (target) target.setAttribute("aria-describedby", h.id);
    kids.push(h);
  }
  return el("div", { class: "c2c-vault-field" }, kids);
}

export function textField({ label, value = "", placeholder = "", autocomplete = "off", hint, maxLength } = {}) {
  const input = el("input", { type: "text", class: "c2c-vault-input", value, placeholder, autocomplete,
    spellcheck: false, maxLength: maxLength || null });
  return { el: field({ label, hint, control: input }), input };
}

export function textArea({ label, placeholder = "", rows = 3, hint } = {}) {
  const input = el("textarea", { class: "c2c-vault-textarea", placeholder, rows, spellcheck: false });
  return { el: field({ label, hint, control: input }), input };
}

const PW_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789!@#$%&*-_=+";
function generatePassword(len = 20) {
  const buf = new Uint32Array(len);
  crypto.getRandomValues(buf);
  let out = "";
  for (let i = 0; i < len; i++) out += PW_ALPHABET[buf[i] % PW_ALPHABET.length];
  return out;
}

/** Password input with Show / Hide, an optional strength meter and an optional Generate. */
export function passwordField({ label, autocomplete = "new-password", strength = false, generate = false, hint,
  placeholder = "", onGenerate } = {}) {
  ensureVaultStyles();
  const input = el("input", { type: "password", class: "c2c-vault-input", autocomplete, placeholder, spellcheck: false });
  const toggle = vbutton("Show", { kind: "text", small: true, ariaLabel: `Show ${label.toLowerCase()}` });
  toggle.setAttribute("aria-pressed", "false");
  toggle.addEventListener("click", () => {
    const show = input.type === "password";
    input.type = show ? "text" : "password";
    if (show) input.classList.add("c2c-vault-secret"); else input.classList.remove("c2c-vault-secret");
    toggle.setAttribute("aria-pressed", show ? "true" : "false");
    toggle.querySelector(".c2c-vault-btn__label").textContent = show ? "Hide" : "Show";
    toggle.setAttribute("aria-label", `${show ? "Hide" : "Show"} ${label.toLowerCase()}`);
  });
  const wrap = el("div", { class: "c2c-vault-inputwrap" }, [input, toggle]);
  const extra = [];
  let meter = null;
  if (strength) {
    const bars = el("div", { class: "c2c-vault-meter__bars", "aria-hidden": "true" },
      [0, 1, 2, 3].map(() => el("span")));
    const txt = el("span", { class: "c2c-vault-meter__text" });
    meter = el("div", { class: "c2c-vault-meter" }, [bars, txt]);
    const paint = () => {
      const s = passwordStrength(input.value);
      const col = !s.ok ? "var(--cu-danger)" : s.score <= 1 ? "var(--cu-warn)" : "var(--cu-ok)";
      [...bars.children].forEach((b, i) => {
        b.style.background = input.value && i < Math.max(1, s.score) ? col : "";
      });
      txt.textContent = s.label ? `Strength: ${s.label}` : "At least 8 characters";
      txt.style.color = !input.value ? "" : !s.ok ? "var(--cu-danger)" : "";
    };
    input.addEventListener("input", paint);
    paint();
  }
  const genOut = el("div");
  if (generate) {
    const gen = vbutton("Generate", { kind: "text", small: true, ariaLabel: "Generate a strong password" });
    gen.title = "Generate a strong password";
    gen.addEventListener("click", () => {
      const pw = generatePassword(20);
      input.value = pw;
      input.dispatchEvent(new Event("input", { bubbles: true }));
      genOut.replaceChildren(copyRow({ label: "generated password", value: pw }),
        el("p", { class: "c2c-vault-dim", text: "Shown once. Store it in your password manager before you lock." }));
      onGenerate?.(pw);
    });
    extra.push(el("div", { class: "c2c-vault-row", style: { justifyContent: "space-between", flexWrap: "wrap" } },
      [meter, gen]));
  } else if (meter) extra.push(meter);
  const f = field({ label, hint, control: wrap });
  if (extra.length) f.append(...extra);
  f.append(genOut);
  return { el: f, input, value: () => input.value, clear: () => { input.value = ""; genOut.replaceChildren();
    input.dispatchEvent(new Event("input")); }, setValue: (v) => { input.value = v; input.dispatchEvent(new Event("input")); } };
}

/** The authenticator code: one input + a bar showing when the code changes. */
export function codeField({ label = "Authenticator code", period = 30, allowRecovery = false, hint } = {}) {
  ensureVaultStyles();
  // inputmode stays "numeric" (the common case is the 6-digit code); allowRecovery only lets a recovery code fit
  const input = el("input", { type: "text", class: "c2c-vault-input c2c-vault-code", autocomplete: "one-time-code",
    maxLength: allowRecovery ? 11 : 6, spellcheck: false, placeholder: allowRecovery ? "6-digit code" : "6 digits" });
  input.setAttribute("inputmode", "numeric");
  const fill = el("span");
  const txt = el("span");
  const clock = el("div", { class: "c2c-vault-clock" }, [el("div", { class: "c2c-vault-clock__bar", "aria-hidden": "true" },
    [fill]), txt]);
  const tick = () => {
    const s = secondsLeft(Date.now(), period);
    fill.style.width = `${(s / period) * 100}%`;
    txt.textContent = `New code in ${s} s`;
  };
  tick();
  // c2c-allow-interval: a 1 s countdown only while an auth dialog is open; destroy() (dialog cleanup) clears it
  const timer = setInterval(tick, 1000); // c2c-allow-interval: cleared by destroy()
  const f = field({ label, hint, control: input });
  f.append(clock);
  return { el: f, input, value: () => input.value.replace(/\s/g, ""), clear: () => { input.value = ""; },
    destroy: () => clearInterval(timer) };
}

/** An APG radio group of cards. Disabled options are left out (say why elsewhere, never a dead control). */
export function choiceCards({ label, options, value, onChange }) {
  ensureVaultStyles();
  const opts = options.filter((o) => !o.disabled);
  let cur = opts.some((o) => o.id === value) ? value : opts[0]?.id;
  const group = el("div", { class: "c2c-vault-choices", role: "radiogroup", "aria-label": label });
  const cards = opts.map((o) => {
    const tagCls = o.tone === "warn" ? " c2c-vault-tag--warn" : o.tone === "ok" ? " c2c-vault-tag--ok" : "";
    const c = el("div", { class: "c2c-vault-choice", role: "radio", "data-id": o.id }, [
      el("span", { class: "c2c-vault-choice__dot", "aria-hidden": "true" }),
      el("span", { class: "c2c-vault-choice__title" }, [o.title, o.tag ? el("span", { class: `c2c-vault-tag${tagCls}`,
        text: o.tag }) : null]),
      el("span", { class: "c2c-vault-choice__sum", text: o.summary || "" })]);
    c.addEventListener("click", () => set(o.id, true));
    return c;
  });
  const sync = () => cards.forEach((c, i) => {
    const on = opts[i].id === cur;
    c.setAttribute("aria-checked", on ? "true" : "false");
    c.tabIndex = on ? 0 : -1;
  });
  function set(id, fire) {
    if (!opts.some((o) => o.id === id)) return;
    const changed = id !== cur;
    cur = id;
    sync();
    if (fire && changed) onChange?.(id);
  }
  group.addEventListener("keydown", (e) => {
    const i = opts.findIndex((o) => o.id === cur);
    let j = null;
    if (e.key === "ArrowDown" || e.key === "ArrowRight") j = (i + 1) % opts.length;
    else if (e.key === "ArrowUp" || e.key === "ArrowLeft") j = (i - 1 + opts.length) % opts.length;
    else if (e.key === " " || e.key === "Enter") { e.preventDefault(); return; }
    if (j == null) return;
    e.preventDefault();
    set(opts[j].id, true);
    cards[j].focus();
  });
  group.append(...cards);
  sync();
  return { el: el("div", { class: "c2c-vault-field" }, [el("div", { class: "c2c-vault-label", text: label }), group]),
    value: () => cur, set: (id) => set(id, true) };
}

export function factList(pairs) {
  ensureVaultStyles();
  const dl = el("dl", { class: "c2c-vault-facts" });
  for (const [k, v] of pairs) if (v) dl.append(el("dt", { text: k }), el("dd", { text: v }));
  return dl;
}

/** A notice block. Errors and warnings are announced (role=alert). */
export function noticeEl(tone, text) {
  ensureVaultStyles();
  return el("div", { class: `c2c-vault-notice c2c-vault-notice--${tone}`,
    role: tone === "danger" || tone === "warn" ? "alert" : "status", text });
}

export function copyRow({ label, value, display, mono = true } = {}) {
  ensureVaultStyles();
  const val = el("div", { class: "c2c-vault-copy__val", text: display ?? value, title: value });
  if (!mono) val.style.fontFamily = "inherit";
  const b = vbutton("Copy", { small: true, ariaLabel: `Copy ${label}` });
  let t = null;
  b.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(String(value ?? ""));
      b.querySelector(".c2c-vault-btn__label").textContent = "Copied";
      clearTimeout(t);
      t = setTimeout(() => { b.querySelector(".c2c-vault-btn__label").textContent = "Copy"; }, 1500);
    } catch (_) {
      const r = document.createRange();
      r.selectNodeContents(val);
      const sel = getSelection();
      sel.removeAllRanges();
      sel.addRange(r);
    }
  });
  return el("div", { class: "c2c-vault-copy" }, [val, b]);
}

/** The selection capsule: these nodes -> one locked node. */
export function capsule({ chips = [], more = 0, inputs = 0, outputs = 0 }) {
  ensureVaultStyles();
  const list = el("div", { class: "c2c-vault-chips" }, chips.map((t) => el("span", { class: "c2c-vault-chip", text: t,
    title: t })));
  if (more > 0) list.append(el("span", { class: "c2c-vault-chip c2c-vault-chip--more", text: `+${more} more` }));
  const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
  return el("div", { class: "c2c-vault-capsule" }, [
    el("div", { class: "c2c-vault-capsule__flow" }, [list, arrowIcon(), sealGlyph({ size: 28 })]),
    el("div", { class: "c2c-vault-capsule__meta", text: `${plural(inputs, "input")} · ${plural(outputs, "output")} ` +
      "on the vault node" })]);
}

export function disclosure(title, { open = false } = {}) {
  ensureVaultStyles();
  const body = el("div", { class: "c2c-vault-disc__body" });
  const d = el("details", { class: "c2c-vault-disc" }, [el("summary", { text: title }), body]);
  d.open = !!open;
  return { el: d, body };
}

export function switchRow(label, { checked = false, onChange } = {}) {
  ensureVaultStyles();
  const b = el("button", { type: "button", class: "c2c-vault-switch", role: "switch", "aria-checked": checked ? "true" : "false" },
    [el("span", { class: "c2c-vault-switch__track", "aria-hidden": "true" }), el("span", { text: label })]);
  let on = !!checked;
  b.addEventListener("click", () => { on = !on; b.setAttribute("aria-checked", on ? "true" : "false"); onChange?.(on); });
  return { el: b, checked: () => on };
}

export function stepSection(n, title, ...children) {
  ensureVaultStyles();
  return el("section", { class: "c2c-vault-step" }, [
    el("span", { class: "c2c-vault-step__n", text: String(n), "aria-hidden": "true" }),
    el("h3", { class: "c2c-vault-step__title", text: title }),
    el("div", { class: "c2c-vault-step__body" }, children.filter(Boolean))]);
}

export function block(title, hint, ...children) {
  ensureVaultStyles();
  return el("section", { class: "c2c-vault-block" }, [el("h3", { class: "c2c-vault-block__title", text: title }),
    hint ? el("p", { class: "c2c-vault-block__hint", text: hint }) : null, ...children.filter(Boolean)]);
}

/** The vault node's body: seal + what it is + run state + one action. Display only. */
export function statusCard() {
  ensureVaultStyles();
  const sealHost = el("span");
  const title = el("div", { class: "c2c-vault-card__title" });
  const detail = el("div", { class: "c2c-vault-card__detail" });
  const meta = el("div", { class: "c2c-vault-card__meta" });
  const actHost = el("span");
  const root = el("div", { class: "c2c-ui c2c-vault c2c-vault-card" }, [sealHost,
    el("div", { class: "c2c-vault-card__text" }, [title, detail, meta]), actHost]);
  for (const ev of ["pointerdown", "mousedown"]) actHost.addEventListener(ev, (e) => e.stopPropagation());
  let lastTone = null;
  return {
    el: root,
    update({ tone = "accent", title: t = "", detail: d = "", meta: m = "", metaTone = "", action = null } = {}) {
      if (tone !== lastTone) { sealHost.replaceChildren(sealGlyph({ size: 26, tone })); lastTone = tone; }
      title.textContent = t;
      title.className = `c2c-vault-card__title${tone === "warn" ? " c2c-vault-card__title--warn" : ""}`;
      detail.textContent = d;
      detail.style.display = d ? "" : "none";
      meta.textContent = m;
      meta.className = `c2c-vault-card__meta${metaTone === "ok" ? " c2c-vault-card__meta--ok" : ""}`;
      meta.style.display = m ? "" : "none";
      actHost.replaceChildren(action ? vbutton(action.label, { small: true, onClick: (e) => { e.stopPropagation();
        action.onClick?.(); }, ariaLabel: action.ariaLabel }) : "");
    },
  };
}
