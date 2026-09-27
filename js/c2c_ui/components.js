/**
 * c2c_ui/components.js — plain-DOM UI primitives for C2C nodes and editors.
 *
 * License: Apache-2.0
 */

import { ensureStyles } from "./theme.js";
import { wolfMark } from "./wolf.js";

function _focusable(el) {
    el.classList.add("c2c-ui-focusable");
    return el;
}

/**
 * @param {Array<{value: string, label: string}>} items
 * @param {{value?: string, onChange?: (v: string) => void, columns?: number}} [opts]
 */
export function pillBar(items, opts = {}) {
    ensureStyles();
    const { value, onChange, columns } = opts;
    const cols = columns || items.length;
    const bar = document.createElement("div");
    bar.className = "c2c-ui-pillbar";
    bar.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;

    let current = value ?? items[0]?.value;

    const sync = () => {
        for (const btn of bar.querySelectorAll(".c2c-ui-pill")) {
            const active = btn.dataset.value === current;
            btn.classList.toggle("c2c-ui-pill--active", active);
            btn.setAttribute("aria-pressed", active ? "true" : "false");
        }
    };

    for (const item of items) {
        const btn = _focusable(document.createElement("button"));
        btn.type = "button";
        btn.className = "c2c-ui-pill";
        btn.dataset.value = item.value;
        btn.textContent = item.label;
        btn.addEventListener("click", () => {
            current = item.value;
            sync();
            onChange?.(item.value);
        });
        bar.appendChild(btn);
    }
    sync();
    return bar;
}

/**
 * @param {Array<{label: string, onClick?: () => void, disabled?: boolean}>} buttons
 */
export function actionRow(buttons) {
    ensureStyles();
    const row = document.createElement("div");
    row.className = "c2c-ui-action-row";
    for (const cfg of buttons) {
        row.appendChild(button(cfg.label, { onClick: cfg.onClick, disabled: cfg.disabled, primary: false }));
    }
    return row;
}

/**
 * @param {string} label
 * @param {{primary?: boolean, danger?: boolean, icon?: HTMLElement|string, onClick?: () => void, disabled?: boolean}} [opts]
 */
export function button(label, opts = {}) {
    ensureStyles();
    const btn = _focusable(document.createElement("button"));
    btn.type = "button";
    btn.className = "c2c-ui-btn";
    if (opts.primary) btn.classList.add("c2c-ui-btn--primary");
    if (opts.danger) btn.classList.add("c2c-ui-btn--danger");
    if (!opts.primary && !opts.danger) btn.classList.add("c2c-ui-btn--outline");
    if (opts.block) btn.classList.add("c2c-ui-btn--block");
    if (opts.disabled) btn.disabled = true;
    if (opts.icon) {
        if (typeof opts.icon === "string") {
            const span = document.createElement("span");
            span.textContent = opts.icon;
            btn.appendChild(span);
        } else {
            btn.appendChild(opts.icon);
        }
    }
    const text = document.createElement("span");
    text.textContent = label;
    btn.appendChild(text);
    if (opts.onClick) btn.addEventListener("click", opts.onClick);
    return btn;
}

/**
 * @param {string} title
 * @param  {...(HTMLElement|string)} children
 */
export function section(title, ...children) {
    ensureStyles();
    const wrap = document.createElement("div");
    wrap.className = "c2c-ui-section";
    const hdr = document.createElement("p");
    hdr.className = "c2c-ui-section__title";
    hdr.textContent = title;
    const body = document.createElement("div");
    body.className = "c2c-ui-section__body";
    for (const child of children) {
        if (typeof child === "string") {
            const p = document.createElement("p");
            p.textContent = child;
            body.appendChild(p);
        } else if (child) {
            body.appendChild(child);
        }
    }
    wrap.appendChild(hdr);
    wrap.appendChild(body);
    return wrap;
}

/**
 * @param {string} label
 * @param {{min: number, max: number, step?: number, value: number, onChange?: (v: number) => void, format?: (v: number) => string}} opts
 */
export function sliderRow(label, opts) {
    ensureStyles();
    const row = document.createElement("div");
    row.className = "c2c-ui-row";
    const lbl = document.createElement("span");
    lbl.className = "c2c-ui-row__label";
    lbl.textContent = label;
    const ctrl = document.createElement("div");
    ctrl.className = "c2c-ui-row__control";

    const step = opts.step ?? 1;
    let val = opts.value;

    const slider = _focusable(document.createElement("input"));
    slider.type = "range";
    slider.className = "c2c-ui-slider";
    slider.min = String(opts.min);
    slider.max = String(opts.max);
    slider.step = String(step);
    slider.value = String(val);

    const num = _focusable(document.createElement("input"));
    num.type = "number";
    num.className = "c2c-ui-num";
    num.min = String(opts.min);
    num.max = String(opts.max);
    num.step = String(step);

    // The number box always holds the raw number (a formatted "80%" is not a
    // valid <input type=number> value and renders blank); `format` is spoken
    // by screen readers and shown as the tooltip instead.
    const fmt = opts.format || ((v) => String(v));
    const sync = (v) => {
        val = Math.min(opts.max, Math.max(opts.min, v));
        slider.value = String(val);
        num.value = String(val);
        slider.setAttribute("aria-valuetext", fmt(val));
        num.title = fmt(val);
        opts.onChange?.(val);
    };

    slider.addEventListener("input", () => sync(parseFloat(slider.value)));
    num.addEventListener("change", () => sync(parseFloat(num.value) || opts.min));
    sync(val);

    ctrl.appendChild(slider);
    ctrl.appendChild(num);
    row.appendChild(lbl);
    row.appendChild(ctrl);
    return row;
}

/**
 * @param {string} label
 * @param {{options: Array<{value: string, label: string}>, value?: string, onChange?: (v: string) => void}} opts
 */
export function selectRow(label, opts) {
    ensureStyles();
    const row = document.createElement("div");
    row.className = "c2c-ui-row";
    const lbl = document.createElement("span");
    lbl.className = "c2c-ui-row__label";
    lbl.textContent = label;
    const ctrl = document.createElement("div");
    ctrl.className = "c2c-ui-row__control";
    const sel = _focusable(document.createElement("select"));
    sel.className = "c2c-ui-select";
    for (const o of opts.options) {
        const opt = document.createElement("option");
        opt.value = o.value;
        opt.textContent = o.label;
        sel.appendChild(opt);
    }
    if (opts.value != null) sel.value = opts.value;
    sel.addEventListener("change", () => opts.onChange?.(sel.value));
    ctrl.appendChild(sel);
    row.appendChild(lbl);
    row.appendChild(ctrl);
    return row;
}

/**
 * @param {string} label
 * @param {{value?: string, onChange?: (hex: string) => void}} opts
 */
export function colorRow(label, opts = {}) {
    ensureStyles();
    const row = document.createElement("div");
    row.className = "c2c-ui-row";
    const lbl = document.createElement("span");
    lbl.className = "c2c-ui-row__label";
    lbl.textContent = label;
    const ctrl = document.createElement("div");
    ctrl.className = "c2c-ui-row__control";
    const inp = _focusable(document.createElement("input"));
    inp.type = "color";
    inp.className = "c2c-ui-color";
    inp.value = opts.value || "#b494ff";
    inp.addEventListener("input", () => opts.onChange?.(inp.value));
    ctrl.appendChild(inp);
    row.appendChild(lbl);
    row.appendChild(ctrl);
    return row;
}

/**
 * @param {string} label
 * @param {{value?: boolean, onChange?: (on: boolean) => void}} opts
 */
export function toggleRow(label, opts = {}) {
    ensureStyles();
    const row = document.createElement("div");
    row.className = "c2c-ui-row";
    const lbl = document.createElement("span");
    lbl.className = "c2c-ui-row__label";
    lbl.textContent = label;
    const ctrl = document.createElement("div");
    ctrl.className = "c2c-ui-row__control";
    let on = !!opts.value;
    const btn = _focusable(document.createElement("button"));
    btn.type = "button";
    btn.className = "c2c-ui-toggle";
    btn.setAttribute("role", "switch");
    const knob = document.createElement("span");
    knob.className = "c2c-ui-toggle__knob";
    btn.appendChild(knob);

    const sync = () => {
        btn.classList.toggle("c2c-ui-toggle--on", on);
        btn.setAttribute("aria-checked", on ? "true" : "false");
    };
    btn.addEventListener("click", () => {
        on = !on;
        sync();
        opts.onChange?.(on);
    });
    sync();
    ctrl.appendChild(btn);
    row.appendChild(lbl);
    row.appendChild(ctrl);
    return row;
}

/**
 * @param {Array<{value: string, label: string, icon?: HTMLElement|string}>} tools
 * @param {{columns?: number, value?: string, onChange?: (v: string) => void}} [opts]
 */
export function toolGrid(tools, opts = {}) {
    ensureStyles();
    const cols = opts.columns || 3;
    const grid = document.createElement("div");
    grid.className = "c2c-ui-toolgrid";
    grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
    let current = opts.value ?? tools[0]?.value;

    const sync = () => {
        for (const t of grid.querySelectorAll(".c2c-ui-tool")) {
            t.classList.toggle("c2c-ui-tool--active", t.dataset.value === current);
        }
    };

    for (const tool of tools) {
        const btn = _focusable(document.createElement("button"));
        btn.type = "button";
        btn.className = "c2c-ui-tool";
        btn.dataset.value = tool.value;
        if (tool.icon) {
            const icon = document.createElement("span");
            icon.className = "c2c-ui-tool__icon";
            if (typeof tool.icon === "string") icon.textContent = tool.icon;
            else icon.appendChild(tool.icon);
            btn.appendChild(icon);
        }
        const lbl = document.createElement("span");
        lbl.textContent = tool.label;
        btn.appendChild(lbl);
        btn.addEventListener("click", () => {
            current = tool.value;
            sync();
            opts.onChange?.(tool.value);
        });
        grid.appendChild(btn);
    }
    sync();
    return grid;
}

/**
 * @param {{onOut?: () => void, onFit?: () => void, onIn?: () => void, getLabel?: () => string}} [opts]
 */
export function zoomBar(opts = {}) {
    ensureStyles();
    const bar = document.createElement("div");
    bar.className = "c2c-ui-zoombar";
    const outBtn = button("\u2212", { onClick: opts.onOut });
    const label = document.createElement("span");
    label.className = "c2c-ui-zoombar__label";
    label.textContent = opts.getLabel?.() || "Fit 100%";
    const fitBtn = button("Fit", { onClick: opts.onFit });
    const inBtn = button("+", { onClick: opts.onIn });
    bar.appendChild(outBtn);
    bar.appendChild(fitBtn);
    bar.appendChild(label);
    bar.appendChild(inBtn);
    return bar;
}

/**
 * @param {{title?: string, hint?: string, glyph?: HTMLElement}} [opts]
 */
export function emptyState(opts = {}) {
    ensureStyles();
    const wrap = document.createElement("div");
    wrap.className = "c2c-ui-empty";
    const markWrap = document.createElement("div");
    markWrap.className = "c2c-ui-empty__mark";
    markWrap.appendChild(opts.glyph || wolfMark(44));
    wrap.appendChild(markWrap);
    if (opts.title) {
        const title = document.createElement("p");
        title.className = "c2c-ui-empty__title";
        title.textContent = opts.title;
        wrap.appendChild(title);
    }
    const brand = document.createElement("p");
    brand.className = "c2c-ui-empty__brand";
    brand.textContent = "C2C";
    wrap.appendChild(brand);
    if (opts.hint) {
        const hint = document.createElement("p");
        hint.className = "c2c-ui-empty__hint";
        hint.textContent = opts.hint;
        wrap.appendChild(hint);
    }
    return wrap;
}

/**
 * @param {{aspect?: number, empty?: boolean|{title?: string, hint?: string}}} [opts]
 *   `empty` is what the stage says while it has nothing to show: the node's
 *   name and one instruction ("Connect images & run to compare").
 */
export function stage(opts = {}) {
    ensureStyles();
    const el = document.createElement("div");
    el.className = "c2c-ui-stage";
    const viewport = document.createElement("div");
    viewport.className = "c2c-ui-stage__viewport";
    if (opts.aspect) {
        viewport.style.aspectRatio = String(opts.aspect);
    }
    const footer = document.createElement("div");
    footer.className = "c2c-ui-stage__footer";
    footer.textContent = "";
    el.appendChild(viewport);
    el.appendChild(footer);

    const emptyCopy = (opts.empty && typeof opts.empty === "object")
        ? opts.empty : { hint: "No preview yet" };
    if (opts.empty) {
        viewport.appendChild(emptyState(emptyCopy));
    }

    return {
        el,
        setImage(src, w, h, alt = "") {
            viewport.innerHTML = "";
            const img = document.createElement("img");
            img.src = src;
            img.alt = alt;
            viewport.appendChild(img);
            footer.textContent = `${w} \u00D7 ${h}`;
        },
        setCanvas(canvas) {
            viewport.innerHTML = "";
            viewport.appendChild(canvas);
        },
        setEmpty(copy) {
            viewport.innerHTML = "";
            viewport.appendChild(emptyState(copy || emptyCopy));
            footer.textContent = "";
        },
        setFooter(text) {
            footer.textContent = text;
        },
    };
}

/** @returns {{el: HTMLElement, setText: (text: string, kind?: string) => void}} */
export function statusLine() {
    ensureStyles();
    const el = document.createElement("div");
    el.className = "c2c-ui-status";
    return {
        el,
        setText(text, kind = "default") {
            el.textContent = text;
            el.classList.remove("c2c-ui-status--ok", "c2c-ui-status--warn", "c2c-ui-status--danger");
            if (kind === "ok") el.classList.add("c2c-ui-status--ok");
            else if (kind === "warn") el.classList.add("c2c-ui-status--warn");
            else if (kind === "danger") el.classList.add("c2c-ui-status--danger");
        },
    };
}
