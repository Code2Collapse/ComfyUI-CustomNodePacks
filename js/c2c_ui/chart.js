/**
 * c2c_ui/chart.js — shared line chart for C2C node panels.
 *
 * License: Apache-2.0
 */

import { ensureStyles } from "./theme.js";
import { emptyState, button } from "./components.js";
import { canvasBackingScale } from "./nodes2.js";

const TICK_FONT = "10px ui-monospace, 'Cascadia Mono', Consolas, monospace";
const TOP_MARGIN = 14;
const TICK_PAD = 4;
const Y_TITLE_BAND = 14;
const X_TITLE_BAND = 12;
const X_TICK_BAND = 12;
const RIGHT_MIN = 12;

/**
 * @param {number} min
 * @param {number} max
 * @param {number} maxCount
 * @returns {{step: number, ticks: number[], decimals: number}}
 */
export function niceTicks(min, max, maxCount = 5) {
    const lo = Number(min);
    const hi = Number(max);
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
        return { step: 1, ticks: [0], decimals: 0 };
    }
    if (lo === hi) {
        const pad = lo === 0 ? 1 : Math.abs(lo) * 0.1;
        return niceTicks(lo - pad, hi + pad, maxCount);
    }
    // maxCount = roughly how many INTERVALS the axis gets (0..1 at 5 -> 0.2 steps).
    const span = hi - lo;
    const raw = span / Math.max(1, maxCount);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    let step = mag;
    if (norm <= 1) step = mag;
    else if (norm <= 2) step = 2 * mag;
    else if (norm <= 5) step = 5 * mag;
    else step = 10 * mag;
    // Each tick is i*step rounded to the step's precision - repeated
    // addition drifts (0.6000000000000001) and the label would show it.
    const decimals = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
    const first = Math.ceil(lo / step - 1e-9);
    const last = Math.floor(hi / step + 1e-9);
    const ticks = [];
    for (let i = first; i <= last && ticks.length <= 2 * maxCount + 2; i++) {
        ticks.push(Number((i * step).toFixed(decimals)));
    }
    if (!ticks.length) ticks.push(lo, hi);
    return { step, ticks, decimals };
}

function _tickLabel(val, decimals) {
    return Number(val).toFixed(decimals);
}

/**
 * @param {number[]} xs ascending
 * @param {number} x
 * @returns {number}
 */
export function nearestIndex(xs, x) {
    if (!xs || !xs.length) return -1;
    if (x <= xs[0]) return 0;
    const last = xs.length - 1;
    if (x >= xs[last]) return last;
    let lo = 0;
    let hi = last;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (xs[mid] < x) lo = mid + 1;
        else hi = mid;
    }
    if (lo > 0 && Math.abs(xs[lo - 1] - x) <= Math.abs(xs[lo] - x)) return lo - 1;
    return lo;
}

/**
 * @param {string} xLabel
 * @param {number[]} xs
 * @param {Array<{label: string, values: Array<number|null|undefined>}>} series
 * @returns {string}
 */
export function seriesToCSV(xLabel, xs, series) {
    const esc = (v) => {
        const s = v == null ? "" : String(v);
        return s.includes(",") || s.includes('"') ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const header = [xLabel, ...series.map((s) => s.label)].map(esc).join(",");
    const rows = (xs || []).map((x, i) => {
        const cells = [x, ...series.map((s) => {
            const v = s.values?.[i];
            return v == null ? "" : v;
        })];
        return cells.map(esc).join(",");
    });
    return [header, ...rows].join("\n");
}

function _resolveColor(root, tokenOrHex) {
    if (!tokenOrHex) return "#b494ff";
    if (String(tokenOrHex).startsWith("#")) return tokenOrHex;
    const prop = String(tokenOrHex).startsWith("--") ? tokenOrHex : `--${tokenOrHex}`;
    try {
        const v = getComputedStyle(root).getPropertyValue(prop).trim();
        if (v) return v;
    } catch (_e) { /* ignore */ }
    const fallbacks = {
        "--cu-series-1": "#b494ff",
        "--cu-series-2": "#76dccb",
        "--cu-series-3": "#f3d288",
        "--cu-series-4": "#f27a92",
        "--cu-series-5": "#7fd4f2",
        "--cu-series-6": "#8ee09d",
        "--cu-danger": "#f27a92",
        "--cu-edge": "#2a2a57",
        "--cu-ink-dim": "#6f6d9b",
        "--cu-ink-soft": "#bab7db",
        "--cu-sunken": "#0c0d23",
    };
    return fallbacks[prop] || "#b494ff";
}

function _fmt(val, fn) {
    if (fn) return fn(val);
    if (Number.isInteger(val)) return String(val);
    return Number(val).toFixed(3);
}

function _domain(values, pad = 0.05) {
    const nums = values.filter((v) => v != null && Number.isFinite(v));
    if (!nums.length) return { min: 0, max: 1 };
    let min = Math.min(...nums);
    let max = Math.max(...nums);
    if (min === max) {
        const d = min === 0 ? 1 : Math.abs(min) * 0.1;
        min -= d;
        max += d;
    } else {
        const span = (max - min) * pad;
        min -= span;
        max += span;
    }
    return { min, max };
}

function _mapX(x, xMin, xMax, left, width) {
    if (xMax === xMin) return left + width / 2;
    return left + ((x - xMin) / (xMax - xMin)) * width;
}

function _mapY(v, yMin, yMax, top, height) {
    if (yMax === yMin) return top + height / 2;
    return top + height - ((v - yMin) / (yMax - yMin)) * height;
}

function _seriesDomains(series, xs, valuesById, visible) {
    const ySeries = series.filter((s) => s.axis !== "y2" && visible.has(s.id) && s.plot !== false);
    const y2Series = series.filter((s) => s.axis === "y2" && visible.has(s.id) && s.plot !== false);
    const yVals = [];
    for (const s of ySeries) {
        const arr = valuesById[s.id] || [];
        for (const v of arr) if (v != null && Number.isFinite(v)) yVals.push(v);
    }
    const y2Vals = [];
    for (const s of y2Series) {
        const arr = valuesById[s.id] || [];
        for (const v of arr) if (v != null && Number.isFinite(v)) y2Vals.push(v);
    }
    return {
        xDom: _domain(xs, 0),
        yDom: _domain(yVals),
        y2Dom: y2Series.length ? _domain(y2Vals) : null,
        hasY2: y2Series.length > 0,
    };
}

function _computeLayout(ctx, cssW, cssH, state, opts) {
    const { xs, xLabel, yLabel, y2Label } = state;
    const domains = _seriesDomains(state.series, xs, state.valuesById, state.visible);
    const xTicks = niceTicks(domains.xDom.min, domains.xDom.max, 6);
    const yTicks = niceTicks(domains.yDom.min, domains.yDom.max, 5);
    const y2Ticks = domains.y2Dom ? niceTicks(domains.y2Dom.min, domains.y2Dom.max, 5) : null;

    ctx.font = TICK_FONT;
    let left = TICK_PAD;
    let right = domains.hasY2 ? TICK_PAD : RIGHT_MIN;
    let bottom = X_TICK_BAND + TICK_PAD + (xLabel ? X_TITLE_BAND : 0);
    const top = TOP_MARGIN;

    if (yTicks.ticks.length) {
        let maxYW = 0;
        for (const ty of yTicks.ticks) {
            maxYW = Math.max(maxYW, ctx.measureText(_tickLabel(ty, yTicks.decimals)).width);
        }
        left = (yLabel ? Y_TITLE_BAND : 0) + maxYW + TICK_PAD;
    } else if (yLabel) {
        left = Y_TITLE_BAND + TICK_PAD;
    }

    if (domains.hasY2 && y2Ticks?.ticks.length) {
        let maxY2W = 0;
        for (const ty of y2Ticks.ticks) {
            maxY2W = Math.max(maxY2W, ctx.measureText(_tickLabel(ty, y2Ticks.decimals)).width);
        }
        right = (y2Label ? Y_TITLE_BAND : 0) + maxY2W + TICK_PAD;
    } else if (domains.hasY2 && y2Label) {
        right = Y_TITLE_BAND + TICK_PAD;
    }

    const plotLeft = left;
    const plotTop = top;
    const plotW = Math.max(1, cssW - left - right);
    const plotH = Math.max(1, cssH - top - bottom);
    return {
        cssW, cssH, plotLeft, plotTop, plotW, plotH,
        margin: { top, right, bottom, left },
        xTicks, yTicks, y2Ticks, ...domains,
    };
}

function _draw(ctx, root, layout, state) {
    const {
        series, xs, valuesById, visible, thresholds, markers, hoverIdx,
        xFormat, yFormat, y2Format, xLabel, yLabel, y2Label,
    } = state;
    const {
        cssW, cssH, plotLeft, plotTop, plotW, plotH,
        xDom, yDom, y2Dom, xTicks, yTicks, y2Ticks,
    } = layout;
    const sunken = _resolveColor(root, "--cu-sunken");
    const edge = _resolveColor(root, "--cu-edge");
    const inkDim = _resolveColor(root, "--cu-ink-dim");
    const inkSoft = _resolveColor(root, "--cu-ink-soft");
    const danger = _resolveColor(root, "--cu-danger");

    ctx.clearRect(0, 0, cssW, cssH);
    ctx.fillStyle = sunken;
    ctx.fillRect(0, 0, cssW, cssH);

    if (!xs?.length) return;

    ctx.strokeStyle = edge;
    ctx.lineWidth = 1;
    ctx.strokeRect(plotLeft, plotTop, plotW, plotH);

    ctx.font = TICK_FONT;
    ctx.fillStyle = inkDim;
    ctx.strokeStyle = edge;
    ctx.globalAlpha = 0.35;
    for (const tx of xTicks.ticks) {
        const px = _mapX(tx, xDom.min, xDom.max, plotLeft, plotW);
        ctx.beginPath();
        ctx.moveTo(px, plotTop);
        ctx.lineTo(px, plotTop + plotH);
        ctx.stroke();
    }
    for (const ty of yTicks.ticks) {
        const py = _mapY(ty, yDom.min, yDom.max, plotTop, plotH);
        ctx.beginPath();
        ctx.moveTo(plotLeft, py);
        ctx.lineTo(plotLeft + plotW, py);
        ctx.stroke();
    }
    ctx.globalAlpha = 1;

    const xLabelPlaced = [];
    const LABEL_Y1 = plotTop - 2;
    const LABEL_Y2 = plotTop - 14;
    const LABEL_GAP = 4;

    for (const th of thresholds || []) {
        ctx.save();
        ctx.strokeStyle = th.kind === "danger" ? danger : inkSoft;
        ctx.setLineDash([5, 4]);
        if (th.axis === "x") {
            const px = _mapX(th.value, xDom.min, xDom.max, plotLeft, plotW);
            ctx.beginPath();
            ctx.moveTo(px, plotTop);
            ctx.lineTo(px, plotTop + plotH);
            ctx.stroke();
        } else {
            const dom = th.axis === "y2" && y2Dom ? y2Dom : yDom;
            const py = _mapY(th.value, dom.min, dom.max, plotTop, plotH);
            ctx.beginPath();
            ctx.moveTo(plotLeft, py);
            ctx.lineTo(plotLeft + plotW, py);
            ctx.stroke();
            if (th.label) {
                ctx.fillStyle = ctx.strokeStyle;
                ctx.textAlign = "left";
                ctx.textBaseline = "bottom";
                ctx.fillText(th.label, plotLeft + 4, py - 2);
            }
        }
        ctx.setLineDash([]);
        ctx.restore();
    }

    const xLabelItems = (thresholds || [])
        .filter((th) => th.axis === "x" && th.label)
        .map((th) => ({
            px: _mapX(th.value, xDom.min, xDom.max, plotLeft, plotW),
            label: th.label,
            stroke: th.kind === "danger" ? danger : inkSoft,
        }))
        .sort((a, b) => a.px - b.px);

    if (xLabelItems.length) {
        ctx.font = TICK_FONT;
        ctx.textBaseline = "bottom";
        for (let i = 0; i < xLabelItems.length; i++) {
            const item = xLabelItems[i];
            const n = xLabelItems.length;
            let align = "center";
            let textX = item.px;
            if (n >= 2 && i === 0) {
                align = "right";
                textX = item.px - LABEL_GAP;
            } else if (n >= 2 && i === n - 1) {
                align = "left";
                textX = item.px + LABEL_GAP;
            }
            const textW = ctx.measureText(item.label).width;
            let textY = LABEL_Y1;
            let left = align === "center" ? textX - textW / 2
                : (align === "right" ? textX - textW : textX);
            const right = left + textW;
            for (const prev of xLabelPlaced) {
                if (Math.abs(textY - prev.y) < 10 && right > prev.left - LABEL_GAP
                    && left < prev.right + LABEL_GAP) {
                    textY = LABEL_Y2;
                    left = align === "center" ? textX - textW / 2
                        : (align === "right" ? textX - textW : textX);
                    break;
                }
            }
            ctx.fillStyle = item.stroke;
            ctx.textAlign = align;
            ctx.fillText(item.label, textX, textY);
            xLabelPlaced.push({ left, right: left + textW, y: textY });
        }
    }

    ctx.fillStyle = inkDim;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const tx of xTicks.ticks) {
        const px = _mapX(tx, xDom.min, xDom.max, plotLeft, plotW);
        ctx.fillText(_tickLabel(tx, xTicks.decimals), px, plotTop + plotH + TICK_PAD);
    }
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const ty of yTicks.ticks) {
        const py = _mapY(ty, yDom.min, yDom.max, plotTop, plotH);
        ctx.fillText(_tickLabel(ty, yTicks.decimals), plotLeft - TICK_PAD, py);
    }
    if (y2Dom && y2Ticks) {
        ctx.textAlign = "left";
        for (const ty of y2Ticks.ticks) {
            const py = _mapY(ty, y2Dom.min, y2Dom.max, plotTop, plotH);
            ctx.fillText(_tickLabel(ty, y2Ticks.decimals), plotLeft + plotW + TICK_PAD, py);
        }
    }

    if (xLabel) {
        ctx.fillStyle = inkSoft;
        ctx.textAlign = "center";
        ctx.textBaseline = "alphabetic";
        ctx.fillText(xLabel, plotLeft + plotW / 2, cssH - 2);
    }
    if (yLabel) {
        ctx.save();
        ctx.translate(Y_TITLE_BAND / 2, plotTop + plotH / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = "center";
        ctx.fillStyle = inkSoft;
        ctx.fillText(yLabel, 0, 0);
        ctx.restore();
    }
    if (y2Label && y2Dom) {
        ctx.save();
        ctx.translate(cssW - Y_TITLE_BAND / 2, plotTop + plotH / 2);
        ctx.rotate(Math.PI / 2);
        ctx.textAlign = "center";
        ctx.fillStyle = inkSoft;
        ctx.fillText(y2Label, 0, 0);
        ctx.restore();
    }

    const dangerThresholds = (thresholds || []).filter((t) => t.kind === "danger");

    for (const s of series) {
        if (!visible.has(s.id) || s.plot === false) continue;
        const vals = valuesById[s.id] || [];
        const dom = s.axis === "y2" && y2Dom ? y2Dom : yDom;
        const color = _resolveColor(root, s.color || "--cu-series-1");
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 1.5;
        ctx.setLineDash(s.dashed ? [4, 3] : []);
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < xs.length; i++) {
            const v = vals[i];
            if (v == null || !Number.isFinite(v)) continue;
            const px = _mapX(xs[i], xDom.min, xDom.max, plotLeft, plotW);
            const py = _mapY(v, dom.min, dom.max, plotTop, plotH);
            if (!started) { ctx.moveTo(px, py); started = true; }
            else ctx.lineTo(px, py);
        }
        if (started) ctx.stroke();
        ctx.setLineDash([]);

        if (s.points) {
            for (let i = 0; i < xs.length; i++) {
                const v = vals[i];
                if (v == null || !Number.isFinite(v)) continue;
                let ptColor = color;
                for (const th of dangerThresholds) {
                    if ((th.axis || "y") === (s.axis || "y") && v > th.value) {
                        ptColor = danger;
                        break;
                    }
                }
                ctx.fillStyle = ptColor;
                const px = _mapX(xs[i], xDom.min, xDom.max, plotLeft, plotW);
                const py = _mapY(v, dom.min, dom.max, plotTop, plotH);
                ctx.beginPath();
                ctx.arc(px, py, 2.5, 0, Math.PI * 2);
                ctx.fill();
            }
        }
    }

    for (const mk of markers || []) {
        const s = series.find((x) => x.id === mk.seriesId);
        if (!s) continue;
        const vals = valuesById[s.id] || [];
        const idx = nearestIndex(xs, mk.x);
        const v = vals[idx];
        if (v == null || !Number.isFinite(v)) continue;
        const dom = s.axis === "y2" && y2Dom ? y2Dom : yDom;
        const px = _mapX(xs[idx], xDom.min, xDom.max, plotLeft, plotW);
        const py = _mapY(v, dom.min, dom.max, plotTop, plotH);
        const color = _resolveColor(root, s.color || "--cu-series-1");
        ctx.fillStyle = color;
        ctx.strokeStyle = color;
        ctx.beginPath();
        ctx.arc(px, py, 4, 0, Math.PI * 2);
        ctx.fill();
        if (mk.label) {
            ctx.fillStyle = inkSoft;
            ctx.textAlign = "left";
            ctx.textBaseline = "bottom";
            ctx.fillText(mk.label, px + 6, py - 4);
        }
    }

    if (hoverIdx != null && hoverIdx >= 0 && hoverIdx < xs.length) {
        const px = _mapX(xs[hoverIdx], xDom.min, xDom.max, plotLeft, plotW);
        ctx.strokeStyle = inkSoft;
        ctx.globalAlpha = 0.6;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(px, plotTop);
        ctx.lineTo(px, plotTop + plotH);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
    }
}

function _updateReadout(el, xs, idx, series, valuesById, visible, formats) {
    if (idx == null || idx < 0 || !xs?.length) {
        el.hidden = true;
        return;
    }
    el.hidden = false;
    el.innerHTML = "";
    const xRow = document.createElement("div");
    xRow.className = "c2c-ui-chart__readout-row";
    xRow.textContent = `x: ${_fmt(xs[idx], formats.x)}`;
    el.appendChild(xRow);
    for (const s of series) {
        if (!visible.has(s.id)) continue;
        if (s.plot === false && !s.readout) continue;
        const vals = valuesById[s.id] || [];
        const v = vals[idx];
        const row = document.createElement("div");
        row.className = "c2c-ui-chart__readout-row";
        const dot = document.createElement("span");
        dot.className = "c2c-ui-chart__readout-dot";
        dot.style.background = s.color?.startsWith("#") ? s.color : `var(${s.color || "--cu-series-1"})`;
        row.appendChild(dot);
        const txt = document.createElement("span");
        const fmt = s.axis === "y2" ? formats.y2 : formats.y;
        txt.textContent = `${s.label}: ${v == null ? "—" : _fmt(v, fmt)}`;
        row.appendChild(txt);
        el.appendChild(row);
    }
}

/**
 * @param {object} opts
 * @returns {object}
 */
export function lineChart(opts = {}) {
    ensureStyles();

    const series = opts.series || [];
    const visible = new Set(series.map((s) => s.id));
    let xs = [];
    let valuesById = {};
    let thresholds = opts.thresholds ? [...opts.thresholds] : [];
    let markers = [];
    let chartState = "empty";
    let message = "";
    let hoverIdx = null;
    let hoverRaf = 0;
    let resizeObs = null;
    let lastLayout = null;
    const listeners = [];

    const root = document.createElement("div");
    root.className = "c2c-ui c2c-ui-chart";
    if (opts.minHeight) root.style.minHeight = `${opts.minHeight}px`;

    const plotWrap = document.createElement("div");
    plotWrap.className = "c2c-ui-chart__plot-wrap";

    const canvas = document.createElement("canvas");
    canvas.className = "c2c-ui-chart__canvas";
    plotWrap.appendChild(canvas);

    const overlay = document.createElement("div");
    overlay.className = "c2c-ui-chart__overlay";
    plotWrap.appendChild(overlay);

    const readout = document.createElement("div");
    readout.className = "c2c-ui-chart__readout";
    readout.hidden = true;
    plotWrap.appendChild(readout);

    root.appendChild(plotWrap);

    const legend = document.createElement("div");
    legend.className = "c2c-ui-chart__legend";

    const chipWrap = document.createElement("div");
    chipWrap.className = "c2c-ui-chart__chips";
    legend.appendChild(chipWrap);

    const copyBtn = button("Copy CSV", { primary: false });
    copyBtn.classList.add("c2c-ui-chart__copy");
    let copyTimer = 0;
    copyBtn.addEventListener("click", () => {
        const text = api.toCSV();
        const done = () => {
            const label = copyBtn.querySelector("span:last-child") || copyBtn;
            const prev = label.textContent;
            label.textContent = "Copied";
            clearTimeout(copyTimer);
            copyTimer = setTimeout(() => { label.textContent = prev; }, 1500);
        };
        if (navigator.clipboard?.writeText) {
            navigator.clipboard.writeText(text).then(done).catch(() => {
                const ta = document.createElement("textarea");
                ta.value = text;
                document.body.appendChild(ta);
                ta.select();
                try { document.execCommand("copy"); done(); } catch (_e) { /* ignore */ }
                ta.remove();
            });
        } else {
            const ta = document.createElement("textarea");
            ta.value = text;
            document.body.appendChild(ta);
            ta.select();
            try { document.execCommand("copy"); done(); } catch (_e) { /* ignore */ }
            ta.remove();
        }
    });
    legend.appendChild(copyBtn);
    root.appendChild(legend);

    function syncLegend() {
        chipWrap.innerHTML = "";
        for (const s of series) {
            if (s.plot === false) continue;
            const chip = document.createElement("button");
            chip.type = "button";
            chip.className = "c2c-ui-chart__chip c2c-ui-focusable";
            chip.setAttribute("aria-pressed", visible.has(s.id) ? "true" : "false");
            const dot = document.createElement("span");
            dot.className = "c2c-ui-chart__chip-dot";
            dot.style.background = s.color?.startsWith("#") ? s.color : `var(${s.color || "--cu-series-1"})`;
            chip.appendChild(dot);
            const lbl = document.createElement("span");
            lbl.textContent = s.label;
            chip.appendChild(lbl);
            chip.addEventListener("click", () => {
                if (visible.has(s.id)) visible.delete(s.id);
                else visible.add(s.id);
                chip.setAttribute("aria-pressed", visible.has(s.id) ? "true" : "false");
                api.redraw();
            });
            chipWrap.appendChild(chip);
        }
    }
    syncLegend();

    function showOverlay() {
        overlay.innerHTML = "";
        overlay.hidden = false;
        plotWrap.classList.add("c2c-ui-chart__plot-wrap--masked");
        if (chartState === "loading") {
            const spin = document.createElement("div");
            spin.className = "c2c-ui-chart__spinner";
            spin.setAttribute("role", "status");
            spin.setAttribute("aria-label", "Loading");
            overlay.appendChild(spin);
        } else if (chartState === "empty") {
            overlay.appendChild(emptyState(opts.empty || { hint: "No data yet" }));
        } else if (chartState === "error") {
            const err = document.createElement("p");
            err.className = "c2c-ui-chart__error";
            err.textContent = message || "Something went wrong.";
            overlay.appendChild(err);
        }
    }

    function hideOverlay() {
        overlay.hidden = true;
        overlay.innerHTML = "";
        plotWrap.classList.remove("c2c-ui-chart__plot-wrap--masked");
    }

    function redraw() {
        if (chartState !== "ready") {
            showOverlay();
            readout.hidden = true;
            return;
        }
        hideOverlay();
        const cssW = Math.max(1, plotWrap.clientWidth || 1);
        const cssH = Math.max(1, plotWrap.clientHeight || 1);
        const scale = canvasBackingScale(cssW, cssH);
        const bw = Math.max(1, Math.round(cssW * scale));
        const bh = Math.max(1, Math.round(cssH * scale));
        if (canvas.width !== bw || canvas.height !== bh) {
            canvas.width = bw;
            canvas.height = bh;
        }
        canvas.style.width = `${cssW}px`;
        canvas.style.height = `${cssH}px`;
        const ctx = canvas.getContext("2d");
        ctx.setTransform(scale, 0, 0, scale, 0, 0);
        const drawState = {
            series, xs, valuesById, visible, thresholds, markers, hoverIdx,
            xFormat: opts.xFormat,
            yFormat: opts.yFormat,
            y2Format: opts.y2Format,
            xLabel: opts.xLabel,
            yLabel: opts.yLabel,
            y2Label: opts.y2Label,
        };
        lastLayout = _computeLayout(ctx, cssW, cssH, drawState, opts);
        _draw(ctx, root, lastLayout, drawState);
        if (hoverIdx != null && hoverIdx >= 0) {
            _updateReadout(readout, xs, hoverIdx, series, valuesById, visible, {
                x: opts.xFormat,
                y: opts.yFormat,
                y2: opts.y2Format,
            });
        } else {
            readout.hidden = true;
        }
    }

    function pointerToX(clientX) {
        const rect = plotWrap.getBoundingClientRect();
        const cw = plotWrap.clientWidth || rect.width || 1;
        const ratio = rect.width > 0 ? cw / rect.width : 1;
        const xLayout = (clientX - rect.left) * ratio;
        const layout = lastLayout;
        if (!layout) return null;
        const { plotLeft, plotW, xDom } = layout;
        const frac = plotW > 0 ? (xLayout - plotLeft) / plotW : 0;
        if (!xs.length || !xDom) return null;
        return xDom.min + frac * (xDom.max - xDom.min);
    }

    const onMove = (e) => {
        if (chartState !== "ready" || !xs.length) return;
        const x = pointerToX(e.clientX);
        if (x == null) return;
        const idx = nearestIndex(xs, x);
        if (idx === hoverIdx) return;
        hoverIdx = idx;
        if (hoverRaf) return;
        hoverRaf = requestAnimationFrame(() => {
            hoverRaf = 0;
            redraw();
        });
    };
    const onLeave = () => {
        if (hoverIdx == null) return;
        hoverIdx = null;
        if (hoverRaf) {
            cancelAnimationFrame(hoverRaf);
            hoverRaf = 0;
        }
        redraw();
    };
    plotWrap.addEventListener("pointermove", onMove);
    plotWrap.addEventListener("pointerleave", onLeave);
    listeners.push(() => {
        plotWrap.removeEventListener("pointermove", onMove);
        plotWrap.removeEventListener("pointerleave", onLeave);
    });

    if (typeof ResizeObserver !== "undefined") {
        resizeObs = new ResizeObserver(() => redraw());
        resizeObs.observe(root);
    }

    const api = {
        el: root,
        setData(newXs, newValues) {
            xs = newXs || [];
            valuesById = newValues || {};
            if (chartState === "ready") redraw();
        },
        setThresholds(list) {
            thresholds = list || [];
            if (chartState === "ready") redraw();
        },
        setMarkers(list) {
            markers = list || [];
            if (chartState === "ready") redraw();
        },
        setState(state, msg) {
            chartState = state;
            message = msg || "";
            if (state === "ready") redraw();
            else {
                hoverIdx = null;
                showOverlay();
            }
        },
        toCSV() {
            const cols = series.filter((s) => visible.has(s.id) && (s.plot !== false || s.readout));
            return seriesToCSV(
                opts.xLabel || "x",
                xs,
                cols.map((s) => ({ label: s.label, values: valuesById[s.id] || [] })),
            );
        },
        redraw,
        destroy() {
            if (hoverRaf) cancelAnimationFrame(hoverRaf);
            hoverRaf = 0;
            try { resizeObs?.disconnect(); } catch (_e) { /* ignore */ }
            resizeObs = null;
            for (const off of listeners) off();
            listeners.length = 0;
            clearTimeout(copyTimer);
        },
    };

    showOverlay();
    return api;
}
