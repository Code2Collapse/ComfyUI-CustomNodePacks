// _c2c_ui_kit.js — shared LTX-quality design system for C2C/MEC node UIs.
//
// WHY: our node editors used ad-hoc inline styles and read as "canvas
// sketches" next to LTX Director, which ships one consistent stylesheet
// (pill buttons, floating-label panels, popover menus, sub-status pills,
// hover/active transitions, a disciplined neutral-dark palette). This module
// is that stylesheet, injected once and shared by every node, so the whole
// pack looks like one finished pro tool.
//
// USAGE:
//   import { ensureC2CKit } from "./_c2c_ui_kit.js";
//   ensureC2CKit();                       // inject the stylesheet (idempotent)
//   root.classList.add("c2ck");           // opt a subtree into the system
//   btn.className = "c2ck-btn";           // then use the classes below
//
// The functional accent colours a node needs (skeleton=blue, iris=red, …)
// stay the node's own business — the kit only styles the CHROME.
//
// License: Apache-2.0

const KIT_ID = "c2ck-styles";

const CSS = `
.c2ck{--k-bg:var(--c2c-bg);--k-panel:var(--c2c-panelBg);--k-panel2:var(--c2c-surface0);--k-line:var(--c2c-border);--k-line2:var(--c2c-surface1);
  --k-fg:var(--c2c-fg);--k-dim:var(--c2c-dim);--k-dim2:var(--c2c-sub);--k-acc:var(--c2c-blue);
  font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;color:var(--k-fg);box-sizing:border-box;}
.c2ck *,.c2ck *::before,.c2ck *::after{box-sizing:border-box;}

/* buttons */
.c2ck-btn{background:var(--k-panel2);color:var(--k-fg);border:1px solid var(--k-line);border-radius:5px;
  padding:5px 11px;font-size:11px;font-weight:500;cursor:pointer;display:inline-flex;align-items:center;gap:5px;
  transition:background .15s ease,border-color .15s ease,transform .05s ease;}
.c2ck-btn:hover:not(:disabled){background:var(--c2c-surface1);border-color:var(--c2c-overlay0);}
.c2ck-btn:active:not(:disabled){transform:translateY(1px);}
.c2ck-btn:disabled{opacity:.45;cursor:not-allowed;}
.c2ck-btn.on{background:color-mix(in srgb,var(--c2c-blue) 18%,var(--c2c-bg2) 82%);border-color:color-mix(in srgb,var(--c2c-blue) 45%,var(--c2c-border) 55%);color:var(--c2c-accentLight);}
.c2ck-btn-danger:hover:not(:disabled){background:color-mix(in srgb,var(--c2c-danger) 22%,var(--c2c-bg2) 78%);border-color:var(--c2c-danger);color:var(--c2c-dangerSoft);}
.c2ck-btn-icon{padding:5px 8px;font-size:12px;justify-content:center;min-width:28px;}
.c2ck-sep{width:1px;align-self:stretch;background:var(--k-line2);margin:2px 3px;}
.c2ck-select{background:var(--k-panel2);color:var(--k-fg);border:1px solid var(--k-line);border-radius:5px;
  padding:4px 6px;font-size:11px;cursor:pointer;}

/* toolbars / status */
.c2ck-toolbar{display:flex;gap:5px;align-items:center;flex-wrap:wrap;}
.c2ck-status{font:11px ui-monospace,monospace;color:var(--k-dim);letter-spacing:.2px;}

/* panels + floating-label fields (the LTX .pr-prompt look) */
.c2ck-panel{background:var(--k-panel);border:1px solid var(--k-line);border-radius:7px;position:relative;box-sizing:border-box;}
.c2ck-fieldwrap{position:relative;width:100%;background:var(--k-panel);border:1px solid var(--k-line);border-radius:7px;
  overflow:hidden;transition:border-color .2s ease;}
.c2ck-fieldwrap:focus-within{border-color:color-mix(in srgb,var(--c2c-blue) 55%,var(--c2c-border) 45%);}
.c2ck-flabel{position:absolute;top:6px;left:9px;font-size:9px;font-weight:700;color:var(--k-dim2);
  text-transform:uppercase;letter-spacing:.6px;pointer-events:none;user-select:none;z-index:2;}
.c2ck-fmeta{position:absolute;top:6px;right:9px;font:9px ui-monospace,monospace;color:var(--k-dim2);pointer-events:none;z-index:2;}
.c2ck-area{width:100%;background:transparent;color:var(--k-fg);border:none;padding:22px 9px 9px;resize:none;
  font-size:12px;line-height:1.45;outline:none;box-sizing:border-box;}
.c2ck-area::placeholder{color:var(--c2c-dim);}
.c2ck-info{background:var(--c2c-bg2);color:var(--c2c-sub);border:1px solid var(--k-line);border-radius:7px;padding:10px 11px;
  font-size:11.5px;line-height:1.6;}
.c2ck-info b,.c2ck-info .hi{color:var(--c2c-fg);font-weight:600;}
.c2ck-itag{display:block;font-size:9px;font-weight:700;color:var(--k-dim2);text-transform:uppercase;
  letter-spacing:.6px;margin-bottom:6px;}

/* sub-status pill (LTX 'Inpaint: ON' / 'Audio: OFF') */
.c2ck-pill{display:inline-flex;align-items:center;gap:4px;font-size:9px;font-weight:600;letter-spacing:.3px;
  padding:2px 7px;border-radius:999px;background:var(--c2c-surface0);border:1px solid var(--k-line2);color:var(--k-dim);}
.c2ck-pill.on{background:color-mix(in srgb,var(--c2c-ok) 14%,var(--c2c-bg2) 86%);border-color:color-mix(in srgb,var(--c2c-ok) 45%,var(--c2c-border) 55%);color:var(--c2c-ok);}
.c2ck-pill.off{background:color-mix(in srgb,var(--c2c-danger) 14%,var(--c2c-bg2) 86%);border-color:color-mix(in srgb,var(--c2c-danger) 45%,var(--c2c-border) 55%);color:var(--c2c-dangerSoft);}

/* toggle chip */
.c2ck-chip{font-size:10.5px;padding:3px 9px;border-radius:999px;cursor:pointer;border:1px solid var(--k-line2);
  background:transparent;color:var(--k-dim);transition:background .12s ease,border-color .12s ease,color .12s ease;}
.c2ck-chip:hover{border-color:var(--c2c-overlay0);}

/* popover menu (build per-open on document.body) */
.c2ck-menu{position:fixed;z-index:2147483000;background:var(--c2c-bg2);border:1px solid var(--c2c-surface1);border-radius:9px;padding:5px;
  box-shadow:0 10px 30px rgba(0,0,0,.6);display:flex;flex-direction:column;gap:1px;min-width:176px;}
.c2ck-menu-head{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--k-dim2);padding:5px 10px 3px;}
.c2ck-menu-btn{display:flex;align-items:center;gap:9px;background:none;border:none;color:var(--k-fg);
  font:12px ui-sans-serif,system-ui;text-align:left;padding:7px 10px;border-radius:6px;cursor:pointer;
  transition:background .12s ease;width:100%;box-sizing:border-box;}
.c2ck-menu-btn:hover:not(:disabled){background:var(--c2c-surface1);}
.c2ck-menu-btn:disabled{opacity:.4;cursor:not-allowed;}
.c2ck-menu-btn .g{width:16px;text-align:center;flex:0 0 auto;color:var(--c2c-dim);}
.c2ck-menu-sep{height:1px;background:var(--c2c-surface1);margin:3px 4px;}

/* range inputs adopt the accent */
.c2ck input[type=range]{accent-color:var(--k-acc);}
`;

/** Inject the shared stylesheet once. Safe to call from every node's setup. */
export function ensureC2CKit() {
    if (typeof document === "undefined") return;
    if (document.getElementById(KIT_ID)) return;
    const el = document.createElement("style");
    el.id = KIT_ID;
    el.textContent = CSS;
    document.head.appendChild(el);
}

/**
 * Build a floating-label popover menu on document.body (eclipse-proof:
 * position:fixed, high z-index, dismiss on outside pointerdown/wheel). Returns
 * a `dismiss()` you should also call from the node's teardown.
 *
 * items: [{ label, glyph?, onPick, disabled?, title? } | { sep:true }]
 */
export function popoverMenu(clientX, clientY, { head, items } = {}) {
    ensureC2CKit();
    const menu = document.createElement("div");
    menu.className = "c2ck-menu";
    if (head) {
        const h = document.createElement("div");
        h.className = "c2ck-menu-head";
        h.textContent = head;
        menu.appendChild(h);
    }
    for (const it of (items || [])) {
        if (it.sep) { const s = document.createElement("div"); s.className = "c2ck-menu-sep"; menu.appendChild(s); continue; }
        const b = document.createElement("button");
        b.className = "c2ck-menu-btn";
        b.innerHTML = `<span class="g">${it.glyph || ""}</span>${it.label}`;
        if (it.disabled) { b.disabled = true; if (it.title) b.title = it.title; }
        else b.addEventListener("pointerdown", (ev) => { ev.preventDefault(); ev.stopPropagation(); dismiss(); try { it.onPick?.(); } catch (_) {} });
        menu.appendChild(b);
    }
    document.body.appendChild(menu);
    const vw = window.innerWidth, vh = window.innerHeight, r = menu.getBoundingClientRect();
    let px = clientX + 4, py = clientY - 6;
    if (px + r.width > vw - 6) px = vw - r.width - 6;
    if (py + r.height > vh - 6) py = vh - r.height - 6;
    menu.style.left = Math.max(6, px) + "px";
    menu.style.top = Math.max(6, py) + "px";
    let onDoc = null;
    function dismiss() {
        try { menu.remove(); } catch (_) {}
        if (onDoc) { document.removeEventListener("pointerdown", onDoc, true); document.removeEventListener("wheel", onDoc, true); onDoc = null; }
    }
    setTimeout(() => {
        onDoc = (ev) => { if (!menu.contains(ev.target)) dismiss(); };
        document.addEventListener("pointerdown", onDoc, true);
        document.addEventListener("wheel", onDoc, true);
    }, 0);
    return { menu, dismiss };
}
