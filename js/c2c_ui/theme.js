/**
 * c2c_ui/theme.js — scoped design tokens and component styles.
 *
 * Call ensureStyles() before mounting any C2C UI. Injects one <style id="c2c-ui-v1">
 * per page (idempotent across packs).
 *
 * License: Apache-2.0
 */

const STYLE_ID = "c2c-ui-v1";

const CSS = `
.c2c-ui,
.c2c-ui-editor {
  --cu-ground: var(--c2c-bg3, #07081a);
  --cu-sunken: var(--c2c-bg2, #0c0d23);
  --cu-panel: var(--c2c-panelBg, #1e1f47);
  --cu-raised: var(--c2c-surface0, #1d1e45);
  --cu-edge: var(--c2c-surface1, #2a2a57);
  --cu-edge-strong: var(--c2c-surface2, #3b3a68);
  --cu-ink: var(--c2c-fg, #e8e6f7);
  --cu-ink-soft: var(--c2c-sub, #bab7db);
  --cu-ink-dim: var(--c2c-dim, #6f6d9b);
  --cu-accent: var(--c2c-mauve, #b494ff);
  --cu-accent-hover: var(--c2c-accentLink, #c0aaff);
  --cu-on-accent: var(--c2c-bg3, #07081a);
  --cu-ok: var(--c2c-ok, #7fe0ab);
  --cu-warn: var(--c2c-yellow, #f3d288);
  --cu-danger: var(--c2c-red, #f27a92);
  font-family: 'Segoe UI', system-ui, sans-serif;
  font-size: 12px;
  color: var(--cu-ink);
  box-sizing: border-box;
}

.c2c-ui *,
.c2c-ui *::before,
.c2c-ui *::after,
.c2c-ui-editor *,
.c2c-ui-editor *::before,
.c2c-ui-editor *::after {
  box-sizing: border-box;
}

.c2c-ui-editor {
  font-size: 13px;
}

/* ── focus ring ─────────────────────────────────────────────────────────── */

.c2c-ui .c2c-ui-focusable:focus {
  outline: none;
}

.c2c-ui .c2c-ui-focusable:focus-visible,
.c2c-ui-editor .c2c-ui-focusable:focus-visible {
  outline: 2px solid var(--cu-accent);
  outline-offset: 2px;
}

/* ── buttons ────────────────────────────────────────────────────────────── */

.c2c-ui .c2c-ui-btn,
.c2c-ui-editor .c2c-ui-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 5px 12px;
  border-radius: 6px;
  border: 1px solid var(--cu-edge);
  background: var(--cu-raised);
  color: var(--cu-ink);
  font: inherit;
  font-size: 12px;
  cursor: pointer;
  transition: border-color 0.12s, background 0.12s;
}

.c2c-ui .c2c-ui-btn--block,
.c2c-ui-editor .c2c-ui-btn--block {
  display: flex;
  width: 100%;
  padding: 7px 12px;
  font-weight: 600;
}

.c2c-ui .c2c-ui-btn:hover:not(:disabled),
.c2c-ui-editor .c2c-ui-btn:hover:not(:disabled) {
  border-color: var(--cu-edge-strong);
  background: var(--cu-panel);
}

.c2c-ui .c2c-ui-btn:disabled,
.c2c-ui-editor .c2c-ui-btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.c2c-ui .c2c-ui-btn--primary,
.c2c-ui-editor .c2c-ui-btn--primary {
  background: var(--cu-accent);
  border-color: var(--cu-accent);
  color: var(--cu-on-accent);
  font-weight: 600;
}

.c2c-ui .c2c-ui-btn--primary:hover:not(:disabled),
.c2c-ui-editor .c2c-ui-btn--primary:hover:not(:disabled) {
  background: var(--cu-accent-hover);
  border-color: var(--cu-accent-hover);
}

.c2c-ui .c2c-ui-btn--danger,
.c2c-ui-editor .c2c-ui-btn--danger {
  border-color: var(--cu-danger);
  color: var(--cu-danger);
}

.c2c-ui .c2c-ui-btn--outline,
.c2c-ui-editor .c2c-ui-btn--outline {
  background: transparent;
}

/* ── pill bar ───────────────────────────────────────────────────────────── */

.c2c-ui .c2c-ui-pillbar,
.c2c-ui-editor .c2c-ui-pillbar {
  display: grid;
  gap: 2px;
  padding: 2px;
  border-radius: 6px;
  background: var(--cu-sunken);
  border: 1px solid var(--cu-edge);
}

.c2c-ui .c2c-ui-pill,
.c2c-ui-editor .c2c-ui-pill {
  padding: 5px 8px;
  border: none;
  border-radius: 4px;
  background: transparent;
  color: var(--cu-ink-soft);
  font: inherit;
  font-size: 11px;
  cursor: pointer;
  text-align: center;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.c2c-ui .c2c-ui-pill:hover:not(:disabled),
.c2c-ui-editor .c2c-ui-pill:hover:not(:disabled) {
  color: var(--cu-ink);
  background: var(--cu-raised);
}

.c2c-ui .c2c-ui-pill--active,
.c2c-ui-editor .c2c-ui-pill--active {
  background: var(--cu-accent);
  color: var(--cu-on-accent);
  font-weight: 600;
}

.c2c-ui .c2c-ui-pill--active:hover,
.c2c-ui-editor .c2c-ui-pill--active:hover {
  background: var(--cu-accent-hover);
  color: var(--cu-on-accent);
}

/* ── action row ─────────────────────────────────────────────────────────── */

.c2c-ui .c2c-ui-action-row,
.c2c-ui-editor .c2c-ui-action-row {
  display: flex;
  gap: 6px;
}

.c2c-ui .c2c-ui-action-row .c2c-ui-btn,
.c2c-ui-editor .c2c-ui-action-row .c2c-ui-btn {
  flex: 1;
}

/* ── section ──────────────────────────────────────────────────────────────── */

.c2c-ui .c2c-ui-section,
.c2c-ui-editor .c2c-ui-section {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.c2c-ui .c2c-ui-section__title,
.c2c-ui-editor .c2c-ui-section__title {
  font-size: 10.5px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--cu-ink-dim);
  margin: 0;
  padding: 0 2px;
}

.c2c-ui .c2c-ui-section__body,
.c2c-ui-editor .c2c-ui-section__body {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

/* Adjacent sections breathe, whatever wrapper holds them. */
.c2c-ui .c2c-ui-section + .c2c-ui-section,
.c2c-ui-editor .c2c-ui-section + .c2c-ui-section {
  margin-top: 18px;
}

/* ── rows (slider, select, color, toggle) ───────────────────────────────── */

.c2c-ui .c2c-ui-row,
.c2c-ui-editor .c2c-ui-row {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 28px;
}

.c2c-ui .c2c-ui-row__label,
.c2c-ui-editor .c2c-ui-row__label {
  flex: 0 0 auto;
  min-width: 72px;
  color: var(--cu-ink-soft);
  font-size: 11px;
}

.c2c-ui .c2c-ui-row__control,
.c2c-ui-editor .c2c-ui-row__control {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}

.c2c-ui .c2c-ui-slider,
.c2c-ui-editor .c2c-ui-slider {
  flex: 1;
  /* Chromium gives range inputs a 129px minimum; without these the number
     box beside it overflowed a 272px sidebar. */
  min-width: 0;
  width: 100%;
  height: 4px;
  accent-color: var(--cu-accent);
  cursor: pointer;
}

.c2c-ui .c2c-ui-num,
.c2c-ui-editor .c2c-ui-num {
  flex: 0 0 auto;
  width: 56px;
  padding: 3px 6px;
  border-radius: 4px;
  border: 1px solid var(--cu-edge);
  background: var(--cu-sunken);
  color: var(--cu-ink);
  font-family: ui-monospace, 'Cascadia Mono', Consolas, monospace;
  font-size: 11px;
  font-variant-numeric: tabular-nums;
  text-align: right;
}

.c2c-ui .c2c-ui-select,
.c2c-ui-editor .c2c-ui-select {
  flex: 1;
  padding: 4px 8px;
  border-radius: 6px;
  border: 1px solid var(--cu-edge);
  background: var(--cu-sunken);
  color: var(--cu-ink);
  font: inherit;
  font-size: 11px;
  cursor: pointer;
}

.c2c-ui .c2c-ui-color,
.c2c-ui-editor .c2c-ui-color {
  width: 32px;
  height: 24px;
  padding: 0;
  border: 1px solid var(--cu-edge);
  border-radius: 4px;
  cursor: pointer;
  background: var(--cu-sunken);
}

.c2c-ui .c2c-ui-toggle,
.c2c-ui-editor .c2c-ui-toggle {
  position: relative;
  width: 36px;
  height: 20px;
  border-radius: 10px;
  border: 1px solid var(--cu-edge);
  background: var(--cu-sunken);
  cursor: pointer;
  padding: 0;
  flex-shrink: 0;
}

.c2c-ui .c2c-ui-toggle__knob,
.c2c-ui-editor .c2c-ui-toggle__knob {
  position: absolute;
  top: 2px;
  left: 2px;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: var(--cu-ink-soft);
  transition: transform 0.15s, background 0.15s;
}

.c2c-ui .c2c-ui-toggle--on,
.c2c-ui-editor .c2c-ui-toggle--on {
  background: var(--cu-accent);
  border-color: var(--cu-accent);
}

.c2c-ui .c2c-ui-toggle--on .c2c-ui-toggle__knob,
.c2c-ui-editor .c2c-ui-toggle--on .c2c-ui-toggle__knob {
  transform: translateX(16px);
  background: var(--cu-on-accent);
}

/* ── tool grid ──────────────────────────────────────────────────────────── */

.c2c-ui .c2c-ui-toolgrid,
.c2c-ui-editor .c2c-ui-toolgrid {
  display: grid;
  gap: 4px;
}

.c2c-ui .c2c-ui-tool,
.c2c-ui-editor .c2c-ui-tool {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 3px;
  padding: 6px 4px;
  border-radius: 6px;
  border: 1px solid var(--cu-edge);
  background: var(--cu-raised);
  color: var(--cu-ink-soft);
  font: inherit;
  font-size: 10px;
  cursor: pointer;
  min-height: 48px;
}

.c2c-ui .c2c-ui-tool:hover:not(:disabled),
.c2c-ui-editor .c2c-ui-tool:hover:not(:disabled) {
  border-color: var(--cu-edge-strong);
  color: var(--cu-ink);
}

.c2c-ui .c2c-ui-tool--active,
.c2c-ui-editor .c2c-ui-tool--active {
  border-color: var(--cu-accent);
  background: var(--cu-panel);
  color: var(--cu-accent);
}

.c2c-ui .c2c-ui-tool__icon,
.c2c-ui-editor .c2c-ui-tool__icon {
  font-size: 16px;
  line-height: 1;
}

/* ── zoom bar ───────────────────────────────────────────────────────────── */

.c2c-ui .c2c-ui-zoombar,
.c2c-ui-editor .c2c-ui-zoombar {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  padding: 3px 4px;
  border-radius: 6px;
  background: var(--cu-panel);
  border: 1px solid var(--cu-edge);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
}

.c2c-ui .c2c-ui-zoombar .c2c-ui-btn,
.c2c-ui-editor .c2c-ui-zoombar .c2c-ui-btn {
  padding: 2px 8px;
  min-width: 28px;
  font-size: 13px;
}

.c2c-ui .c2c-ui-zoombar__label,
.c2c-ui-editor .c2c-ui-zoombar__label {
  padding: 0 6px;
  font-family: ui-monospace, 'Cascadia Mono', Consolas, monospace;
  font-size: 11px;
  font-variant-numeric: tabular-nums;
  color: var(--cu-ink-soft);
  white-space: nowrap;
}

/* ── empty state ────────────────────────────────────────────────────────── */

.c2c-ui .c2c-ui-empty,
.c2c-ui-editor .c2c-ui-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 16px;
  text-align: center;
  color: var(--cu-ink-dim);
}

.c2c-ui .c2c-ui-empty__mark,
.c2c-ui-editor .c2c-ui-empty__mark {
  color: var(--cu-accent);
  margin-bottom: 4px;
  line-height: 0;
}

.c2c-ui .c2c-ui-empty__title,
.c2c-ui-editor .c2c-ui-empty__title {
  font-size: 15px;
  font-weight: 700;
  color: var(--cu-ink);
  margin: 0;
}

.c2c-ui .c2c-ui-empty__brand,
.c2c-ui-editor .c2c-ui-empty__brand {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.12em;
  color: var(--cu-accent);
  margin: 0;
}

.c2c-ui .c2c-ui-empty__hint,
.c2c-ui-editor .c2c-ui-empty__hint {
  font-size: 11px;
  color: var(--cu-ink-dim);
  margin: 0;
  max-width: 240px;
  line-height: 1.4;
}

/* ── stage ──────────────────────────────────────────────────────────────── */

.c2c-ui .c2c-ui-stage,
.c2c-ui-editor .c2c-ui-stage {
  display: flex;
  flex-direction: column;
  border-radius: 8px;
  border: 1px solid var(--cu-edge);
  background: var(--cu-sunken);
  overflow: hidden;
}

.c2c-ui .c2c-ui-stage__viewport,
.c2c-ui-editor .c2c-ui-stage__viewport {
  position: relative;
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 0;
  overflow: hidden;
}

.c2c-ui .c2c-ui-stage__viewport img,
.c2c-ui-editor .c2c-ui-stage__viewport img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  display: block;
}

.c2c-ui .c2c-ui-stage__viewport canvas,
.c2c-ui-editor .c2c-ui-stage__viewport canvas {
  max-width: 100%;
  max-height: 100%;
  display: block;
}

.c2c-ui .c2c-ui-stage__footer,
.c2c-ui-editor .c2c-ui-stage__footer {
  padding: 4px 8px;
  border-top: 1px solid var(--cu-edge);
  font-family: ui-monospace, 'Cascadia Mono', Consolas, monospace;
  font-size: 10px;
  font-variant-numeric: tabular-nums;
  color: var(--cu-ink-dim);
  text-align: center;
}

/* ── status line ────────────────────────────────────────────────────────── */

.c2c-ui .c2c-ui-status,
.c2c-ui-editor .c2c-ui-status {
  padding: 4px 8px;
  font-size: 11px;
  color: var(--cu-ink-soft);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.c2c-ui .c2c-ui-status--ok,
.c2c-ui-editor .c2c-ui-status--ok { color: var(--cu-ok); }

.c2c-ui .c2c-ui-status--warn,
.c2c-ui-editor .c2c-ui-status--warn { color: var(--cu-warn); }

.c2c-ui .c2c-ui-status--danger,
.c2c-ui-editor .c2c-ui-status--danger { color: var(--cu-danger); }

/* ── editor shell ───────────────────────────────────────────────────────── */

.c2c-ui-editor.c2c-ui-editor-overlay {
  position: fixed;
  inset: 0;
  z-index: var(--c2c-z-modal, 10000);
  display: flex;
  flex-direction: column;
  background: var(--cu-ground);
}

.c2c-ui-editor .c2c-ui-editor__header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  background: var(--cu-panel);
  border-bottom: 1px solid var(--cu-edge);
  flex-shrink: 0;
}

.c2c-ui-editor .c2c-ui-editor__brand {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
}

.c2c-ui-editor .c2c-ui-editor__brand-text {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.12em;
  color: var(--cu-accent);
}

.c2c-ui-editor .c2c-ui-editor__title {
  font-size: 15px;
  font-weight: 700;
  color: var(--cu-ink);
  margin: 0;
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.c2c-ui-editor .c2c-ui-editor__header-actions {
  display: flex;
  gap: 4px;
  flex-shrink: 0;
}

.c2c-ui-editor .c2c-ui-editor__body {
  display: flex;
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

.c2c-ui-editor .c2c-ui-editor__left {
  width: 272px;
  flex-shrink: 0;
  background: var(--cu-panel);
  border-right: 1px solid var(--cu-edge);
  overflow-y: auto;
  padding: 14px 12px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.c2c-ui-editor .c2c-ui-editor__centre {
  flex: 1;
  position: relative;
  background: var(--cu-ground);
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 0;
}

.c2c-ui-editor .c2c-ui-editor__right {
  width: 300px;
  flex-shrink: 0;
  background: var(--cu-panel);
  border-left: 1px solid var(--cu-edge);
  overflow-y: auto;
  padding: 14px 12px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.c2c-ui-editor .c2c-ui-editor__footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 12px;
  background: var(--cu-panel);
  border-top: 1px solid var(--cu-edge);
  flex-shrink: 0;
  gap: 12px;
}

.c2c-ui-editor .c2c-ui-editor__hints {
  flex: 1;
  font-size: 11px;
  color: var(--cu-ink-dim);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.c2c-ui-editor .c2c-ui-editor__hints[data-kind="danger"] {
  color: var(--cu-danger);
  white-space: normal;
}

.c2c-ui-editor .c2c-ui-editor__actions {
  display: flex;
  gap: 8px;
  flex-shrink: 0;
}

.c2c-ui-editor .c2c-ui-editor__confirm {
  position: absolute;
  bottom: 52px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 16px;
  border-radius: 8px;
  background: var(--cu-panel);
  border: 1px solid var(--cu-edge);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45);
  z-index: 2;
}

.c2c-ui-editor .c2c-ui-editor__confirm[hidden] {
  display: none;   /* the display:flex above would otherwise beat [hidden] */
}

.c2c-ui-editor .c2c-ui-editor__confirm-text {
  font-size: 12px;
  color: var(--cu-ink-soft);
}
`;

/** Inject scoped C2C UI styles once per page. */
export function ensureStyles() {
    if (typeof document === "undefined") return;
    if (document.getElementById(STYLE_ID)) return;
    const el = document.createElement("style");
    el.id = STYLE_ID;
    el.textContent = CSS;
    document.head.appendChild(el);
}
