// ─────────────────────────────────────────────────────────────────────
// MEC Advanced Paint Canvas — c2c_ui node panel + full-screen editor
// ─────────────────────────────────────────────────────────────────────
//   • Node: compact preview stage + "Open paint editor" (not paintable)
//   • Editor: left tools/brush/canvas, centre workspace + zoom, right output
//   • Left-drag = paint · Right-drag = erase · Wheel = brush size
//   • Pixels serialised as PNG → hidden canvas_data widget on Save / Clear
// ─────────────────────────────────────────────────────────────────────
import { app } from "../../scripts/app.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";
import {
    mountPanel,
    openEditor,
    button,
    actionRow,
    section,
    sliderRow,
    colorRow,
    toolGrid,
    zoomBar,
    stage,
    pillBar,
} from "./c2c_ui/index.js";

const NODE_NAME = "MECAdvancedPaintCanvas";
const UNDO_BYTE_BUDGET = 256_000_000;
// Transparency checkerboard: MID grey (a faint indigo tint to sit in the house
// look), as in every paint app, so black and white paint both read on it -
// ~3:1 for black, ~5:1 for white. It used to be the night ground itself
// (#07081a/#0c0d23), where the default black brush all but vanished.
const CHECKER_A = "#6e6f82";
const CHECKER_B = "#5b5c6f";
const BRUSH_WIDGETS_HIDE = ["brush_color", "brush_opacity", "brush_hardness", "brush_size"];
const MASK_WIDGET_PREFIX = "mask_";

function hexToRgb(hex) {
    if (!hex) return [0, 0, 0];
    let h = hex.trim();
    if (h.startsWith("#")) h = h.slice(1);
    if (h.length === 3) h = h.split("").map((c) => c + c).join("");
    const v = parseInt(h, 16);
    if (isNaN(v)) return [0, 0, 0];
    return [(v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff];
}

function getWidget(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

function hideWidget(w) {
    if (!w) return;
    if (!w.options) w.options = {};
    if (w.element) w.element.style.display = "none";
    w.serializeValue = w.serializeValue || (() => w.value ?? "");
    w.computeSize = () => [0, -4];
    w.type = "hidden";
    w.draw = w.draw || (() => {});
    w.options.hidden = true;
    const el = w.element;
    if (el) {
        el.hidden = true;
        el.style.display = "none";
        const wrapper = el.parentElement;
        if (wrapper && wrapper.classList?.contains("dom-widget")) {
            wrapper.style.display = "none";
        }
    }
}

function readNumOpts(w, defaults) {
    const o = w?.options || {};
    return {
        min: o.min ?? defaults.min,
        max: o.max ?? defaults.max,
        step: o.step ?? defaults.step,
    };
}

function markGraphDirty(node) {
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function setWidgetValue(node, w, value) {
    if (!w) return;
    w.value = value;
    w.callback?.call(w, value);
    markGraphDirty(node);
}

function paintCheckerboard(ctx, w, h, tile = 24) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, w, h);
    ctx.clip();
    for (let y = 0; y < h; y += tile) {
        for (let x = 0; x < w; x += tile) {
            ctx.fillStyle = (((x + y) / tile) % 2 === 0) ? CHECKER_A : CHECKER_B;
            ctx.fillRect(x, y, Math.min(tile, w - x), Math.min(tile, h - y));
        }
    }
    ctx.restore();
}

function undoCapForSize(w, h) {
    const bytes = Math.max(1, w * h * 4);
    return Math.max(3, Math.min(30, Math.floor(UNDO_BYTE_BUDGET / bytes)));
}

function canvasDims(node) {
    const w = +(getWidget(node, "canvas_width")?.value ?? 512);
    const h = +(getWidget(node, "canvas_height")?.value ?? 512);
    return [w, h];
}

function previewDataUrl(ctrl) {
    const [w, h] = ctrl.size;
    const tmp = document.createElement("canvas");
    tmp.width = w;
    tmp.height = h;
    const ctx = tmp.getContext("2d");
    paintCheckerboard(ctx, w, h);
    ctx.drawImage(ctrl.draw, 0, 0);
    return tmp.toDataURL("image/png");
}

function refreshNodeStage(stageApi, ctrl, node) {
    const [w, h] = canvasDims(node);
    const vp = stageApi.el.querySelector(".c2c-ui-stage__viewport");
    if (vp) vp.style.aspectRatio = String(w / h);
    stageApi.setFooter(`${w} \u00D7 ${h}`);
    if (!ctrl.hasInk()) {
        stageApi.setEmpty({
            title: "Advanced Paint Canvas",
            hint: "Open the editor to start painting",
        });
        stageApi.setFooter(`${w} \u00D7 ${h}`);
    } else {
        stageApi.setImage(previewDataUrl(ctrl), w, h, "Paint preview");
    }
}

function defaultRecents(node) {
    if (!node._mecPaintRecents) {
        node._mecPaintRecents = [
            "#000000", "#ffffff", "#b494ff", "#f27a92",
            "#7fe0ab", "#f3d288", "#3b82f6", "#ff6b00",
        ];
    }
    return node._mecPaintRecents;
}

function pushRecentColor(node, hex) {
    const list = defaultRecents(node);
    const i = list.indexOf(hex);
    if (i >= 0) list.splice(i, 1);
    list.unshift(hex);
    while (list.length > 8) list.pop();
}

function buildRecentColorRow(node, onPick) {
    const row = document.createElement("div");
    row.className = "c2c-ui-row";
    row.style.marginTop = "6px";
    const lbl = document.createElement("span");
    lbl.className = "c2c-ui-row__label";
    lbl.textContent = "Recent";
    const ctrl = document.createElement("div");
    ctrl.className = "c2c-ui-row__control";
    ctrl.style.display = "flex";
    ctrl.style.gap = "4px";
    ctrl.style.flexWrap = "wrap";

    const sync = () => {
        ctrl.innerHTML = "";
        for (const hex of defaultRecents(node)) {
            const sw = document.createElement("button");
            sw.type = "button";
            sw.className = "c2c-ui-focusable";
            sw.title = hex;
            sw.style.cssText = `width:22px;height:22px;border-radius:4px;border:1px solid var(--cu-edge,#2a2a57);background:${hex};cursor:pointer;padding:0;`;
            sw.addEventListener("click", () => onPick(hex));
            ctrl.appendChild(sw);
        }
    };
    sync();
    row._syncRecents = sync;
    row.appendChild(lbl);
    row.appendChild(ctrl);
    return row;
}

function createUndoStack(ctrl) {
    let history = [];
    let index = -1;
    let cap = undoCapForSize(ctrl.size[0], ctrl.size[1]);
    let sizeKey = `${ctrl.size[0]}x${ctrl.size[1]}`;

    const trimOldest = () => {
        while (history.length > cap) {
            history.shift();
            index = Math.max(0, index - 1);
        }
    };

    const syncCap = () => {
        cap = undoCapForSize(ctrl.size[0], ctrl.size[1]);
        trimOldest();
    };

    return {
        reset() {
            history = [ctrl.captureSnapshot()];
            index = 0;
            sizeKey = `${ctrl.size[0]}x${ctrl.size[1]}`;
            syncCap();
        },
        onCanvasResize() {
            const key = `${ctrl.size[0]}x${ctrl.size[1]}`;
            if (key !== sizeKey) {
                sizeKey = key;
                this.reset();
                return;
            }
            syncCap();
        },
        push() {
            syncCap();
            history = history.slice(0, index + 1);
            history.push(ctrl.captureSnapshot());
            index = history.length - 1;
            trimOldest();
        },
        undo() {
            if (index <= 0) return false;
            index -= 1;
            ctrl.restoreSnapshot(history[index]);
            return true;
        },
        redo() {
            if (index >= history.length - 1) return false;
            index += 1;
            ctrl.restoreSnapshot(history[index]);
            return true;
        },
        canUndo() { return index > 0; },
        canRedo() { return index < history.length - 1; },
    };
}

class PaintCanvasController {
    constructor(node, opts = {}) {
        this.node = node;
        this.interactive = opts.interactive !== false;
        this._serialiseOnStroke = opts.serialiseOnStroke === true;
        this.size = [512, 512];
        this.dpr = window.devicePixelRatio || 1;
        this.onStrokeEnd = null;
        this.onLoaded = null;
        this._hasInk = false;
        this._down = false;
        this._eraser = false;
        this._toolEraser = false;
        this._last = null;
        this._mouse = null;
        this._serTimer = null;

        this.root = document.createElement("div");
        Object.assign(this.root.style, {
            position: "relative",
            width: `${this.size[0]}px`,
            height: `${this.size[1]}px`,
            overflow: "hidden",
            userSelect: "none",
            touchAction: "none",
            flexShrink: "0",
        });

        this.backdrop = document.createElement("canvas");
        Object.assign(this.backdrop.style, {
            position: "absolute", inset: 0,
            width: "100%", height: "100%",
            pointerEvents: "none",
        });
        this.bctx = this.backdrop.getContext("2d");

        this.draw = document.createElement("canvas");
        this.draw.width = this.size[0];
        this.draw.height = this.size[1];
        Object.assign(this.draw.style, {
            position: "absolute", inset: 0,
            width: "100%", height: "100%",
            cursor: this.interactive ? "crosshair" : "default",
            pointerEvents: this.interactive ? "auto" : "none",
        });
        this.ctx = this.draw.getContext("2d");

        this.cursor = document.createElement("canvas");
        Object.assign(this.cursor.style, {
            position: "absolute", inset: 0,
            width: "100%", height: "100%",
            pointerEvents: "none",
            display: this.interactive ? "block" : "none",
        });
        this.cctx = this.cursor.getContext("2d");

        this._sizeBackdrop(this.size[0], this.size[1]);
        this.root.appendChild(this.backdrop);
        this.root.appendChild(this.draw);
        this.root.appendChild(this.cursor);

        this._onUpBound = (e) => this._onUp(e);
        if (this.interactive) this._bind();
    }

    _bind() {
        const c = this.draw;
        c.addEventListener("contextmenu", (e) => e.preventDefault());
        c.addEventListener("pointerdown", (e) => this._onDown(e));
        c.addEventListener("pointermove", (e) => this._onMove(e));
        // End the stroke on the canvas itself. Inside the full-screen editor
        // the overlay stops pointerup before it reaches window, and a
        // window-only listener never saw it: no undo step was recorded and
        // the brush stayed down, painting on every later mouse move.
        c.addEventListener("pointerup", this._onUpBound);
        c.addEventListener("pointercancel", this._onUpBound);
        c.addEventListener("lostpointercapture", this._onUpBound);
        window.addEventListener("pointerup", this._onUpBound);
        c.addEventListener("pointerleave", () => { this._mouse = null; this._drawCursor(); });
        c.addEventListener("wheel", (e) => this._onWheel(e), { passive: false });
    }

    dispose() {
        window.removeEventListener("pointerup", this._onUpBound);
        if (this._serTimer) clearTimeout(this._serTimer);
    }

    hasInk() { return this._hasInk; }

    setEraser(on) {
        this._toolEraser = !!on;
    }

    _localPos(e) {
        const r = this.draw.getBoundingClientRect();
        const x = ((e.clientX - r.left) / r.width) * this.size[0];
        const y = ((e.clientY - r.top) / r.height) * this.size[1];
        return [x, y];
    }

    _onDown(e) {
        e.preventDefault();
        try { this.draw.setPointerCapture?.(e.pointerId); } catch (_) {}
        this._down = true;
        if (!this._hasInk) {
            this._hasInk = true;
            this._paintBackdrop();
        }
        this._eraser = e.button === 2 || this._toolEraser;
        this._last = this._localPos(e);
        this._stamp(this._last[0], this._last[1]);
    }

    _onMove(e) {
        this._mouse = this._localPos(e);
        if (this._down) {
            this._stroke(this._last, this._mouse);
            this._last = this._mouse;
            this._serialiseSoon();
        }
        this._drawCursor();
    }

    _onUp() {
        if (!this._down) return;
        this._down = false;
        this._serialiseSoon();
        this.onStrokeEnd?.();
    }

    _onWheel(e) {
        e.preventDefault();
        const w = getWidget(this.node, "brush_size");
        if (!w) return;
        const o = readNumOpts(w, { min: 1, max: 500, step: 1 });
        const dir = e.deltaY > 0 ? -2 : 2;
        setWidgetValue(this.node, w, Math.max(o.min, Math.min(o.max, (w.value | 0) + dir)));
        this._drawCursor();
    }

    _brushParams() {
        const size = +(getWidget(this.node, "brush_size")?.value ?? 20);
        const hard = +(getWidget(this.node, "brush_hardness")?.value ?? 0.8);
        const op = +(getWidget(this.node, "brush_opacity")?.value ?? 1.0);
        const col = (getWidget(this.node, "brush_color")?.value ?? "#000000");
        return { size, hard, op, col };
    }

    _stamp(x, y) {
        const { size, hard, op, col } = this._brushParams();
        const r = Math.max(1, size * 0.5);
        const ctx = this.ctx;
        ctx.save();
        ctx.globalCompositeOperation = this._eraser ? "destination-out" : "source-over";
        const [R, G, B] = hexToRgb(col);
        const grad = ctx.createRadialGradient(x, y, 0, x, y, r);
        const inner = Math.max(0, Math.min(1, hard));
        grad.addColorStop(0, `rgba(${R},${G},${B},${op})`);
        grad.addColorStop(inner, `rgba(${R},${G},${B},${op})`);
        grad.addColorStop(1, `rgba(${R},${G},${B},0)`);
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
    }

    _stroke(a, b) {
        const { size } = this._brushParams();
        const r = Math.max(1, size * 0.5);
        const dx = b[0] - a[0], dy = b[1] - a[1];
        const dist = Math.hypot(dx, dy);
        const step = Math.max(1, r * 0.25);
        const n = Math.ceil(dist / step);
        for (let i = 1; i <= n; i++) {
            const t = i / n;
            this._stamp(a[0] + dx * t, a[1] + dy * t);
        }
    }

    _drawCursor() {
        const cv = this.cursor;
        if (!this.interactive) return;
        if (cv.width !== this.size[0] || cv.height !== this.size[1]) {
            cv.width = this.size[0];
            cv.height = this.size[1];
        }
        const ctx = this.cctx;
        ctx.clearRect(0, 0, cv.width, cv.height);
        if (!this._mouse) return;
        const { size, hard } = this._brushParams();
        const r = Math.max(1, size * 0.5);
        const [x, y] = this._mouse;
        ctx.save();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = "rgba(255,255,255,0.95)";
        if (hard >= 0.99) {
            ctx.beginPath();
            ctx.arc(x, y, r, 0, Math.PI * 2);
            ctx.stroke();
        } else {
            ctx.setLineDash([3, 3]);
            for (let i = 0; i < 4; i++) {
                const f = 1.0 - (1.0 - hard) * (i / 4);
                ctx.beginPath();
                ctx.arc(x, y, r * f, 0, Math.PI * 2);
                ctx.stroke();
            }
        }
        ctx.restore();
    }

    _serialiseSoon() {
        if (!this._serialiseOnStroke) return;
        if (this._serTimer) return;
        this._serTimer = setTimeout(() => {
            this._serTimer = null;
            this.serialise();
        }, 60);
    }

    serialise() {
        const w = getWidget(this.node, "canvas_data");
        if (!w) return;
        try {
            w.value = this.draw.toDataURL("image/png");
        } catch (e) {
            console.warn("[MEC paint] serialise failed", e);
        }
    }

    captureSnapshot() {
        return this.ctx.getImageData(0, 0, this.size[0], this.size[1]);
    }

    restoreSnapshot(data) {
        if (!data || data.width !== this.size[0] || data.height !== this.size[1]) return;
        this.ctx.putImageData(data, 0, 0);
        this._hasInk = true;
        this._paintBackdrop();
    }

    copyFrom(other) {
        if (!other) return;
        const [w, h] = other.size;
        if (this.size[0] !== w || this.size[1] !== h) this.setSize(w, h);
        this.ctx.clearRect(0, 0, w, h);
        this.ctx.drawImage(other.draw, 0, 0);
        this._hasInk = other.hasInk();
        this._paintBackdrop();
    }

    setSize(w, h) {
        if (this.size[0] === w && this.size[1] === h) return;
        const tmp = document.createElement("canvas");
        tmp.width = this.draw.width;
        tmp.height = this.draw.height;
        tmp.getContext("2d").drawImage(this.draw, 0, 0);
        this.draw.width = w;
        this.draw.height = h;
        this.cursor.width = w;
        this.cursor.height = h;
        this._sizeBackdrop(w, h);
        this.size = [w, h];
        this.root.style.width = `${w}px`;
        this.root.style.height = `${h}px`;
        this.ctx.drawImage(tmp, 0, 0, w, h);
        this._drawCursor();
    }

    _paintBackdrop() {
        if (!this.bctx) return;
        const w = this.backdrop.width, h = this.backdrop.height;
        if (!w || !h) return;
        paintCheckerboard(this.bctx, w, h);
    }

    _sizeBackdrop(w, h) {
        this.backdrop.width = w;
        this.backdrop.height = h;
        this._paintBackdrop();
    }

    clear() {
        this.ctx.clearRect(0, 0, this.draw.width, this.draw.height);
        this._hasInk = false;
        this._paintBackdrop();
        if (this._serialiseOnStroke) this._serialiseSoon();
    }

    loadFromDataURL(url) {
        if (!url) return;
        const img = new Image();
        img.onload = () => {
            this.ctx.clearRect(0, 0, this.draw.width, this.draw.height);
            this.ctx.drawImage(img, 0, 0, this.draw.width, this.draw.height);
            this._hasInk = true;
            this._synced = true;
            this._paintBackdrop();
            this.onLoaded?.();
        };
        // A saved painting that cannot be decoded stays in canvas_data
        // untouched: _synced stays false, so nothing overwrites it.
        img.onerror = () => { this.onLoaded?.(); };
        img.src = url;
    }
}

function bindSliderFromWidget(node, name, defaults, onExtra) {
    const w = getWidget(node, name);
    if (!w) return null;
    const o = readNumOpts(w, defaults);
    return sliderRow(w.name.replace(/_/g, " "), {
        min: o.min, max: o.max, step: o.step,
        value: +w.value,
        format: defaults.format,
        onChange: (v) => {
            setWidgetValue(node, w, v);
            onExtra?.(v);
        },
    });
}

function buildEditorLeft(node, editorCtrl, editorState, undoStack, recentRow) {
    const wrap = document.createElement("div");

    const tools = toolGrid([
        { value: "brush", label: "Brush", icon: "\u270F" },
        { value: "eraser", label: "Eraser", icon: "\u232B" },
    ], {
        columns: 2,
        value: editorState.tool,
        onChange: (v) => {
            editorState.tool = v;
            editorCtrl.setEraser(v === "eraser");
        },
    });
    wrap.appendChild(section("Tools", tools));

    const brushBody = document.createElement("div");
    const sizeRow = bindSliderFromWidget(node, "brush_size",
        { min: 1, max: 500, step: 1 },
        () => editorCtrl._drawCursor());
    const hardRow = bindSliderFromWidget(node, "brush_hardness",
        { min: 0, max: 1, step: 0.01 },
        () => editorCtrl._drawCursor());
    const opRow = bindSliderFromWidget(node, "brush_opacity",
        { min: 0, max: 1, step: 0.01, format: (v) => `${Math.round(v * 100)}%` });
    const colW = getWidget(node, "brush_color");
    const colRow = colorRow("Color", {
        value: colW?.value || "#000000",
        onChange: (hex) => {
            if (colW) setWidgetValue(node, colW, hex);
            pushRecentColor(node, hex);
            recentRow._syncRecents?.();
        },
    });
    if (sizeRow) brushBody.appendChild(sizeRow);
    if (hardRow) brushBody.appendChild(hardRow);
    if (opRow) brushBody.appendChild(opRow);
    brushBody.appendChild(colRow);
    brushBody.appendChild(recentRow);
    wrap.appendChild(section("Brush", brushBody));

    const canvasBody = document.createElement("div");
    const onDimChange = () => {
        const [w, h] = canvasDims(node);
        editorCtrl.setSize(w, h);
        undoStack.onCanvasResize();
        editorState.onFit?.();
    };
    const wRow = bindSliderFromWidget(node, "canvas_width",
        { min: 64, max: 4096, step: 8 }, onDimChange);
    const hRow = bindSliderFromWidget(node, "canvas_height",
        { min: 64, max: 4096, step: 8 }, onDimChange);
    if (wRow) canvasBody.appendChild(wRow);
    if (hRow) canvasBody.appendChild(hRow);
    wrap.appendChild(section("Canvas", canvasBody));

    return wrap;
}

function buildEditorRight(node) {
    const wrap = document.createElement("div");
    const body = document.createElement("div");

    const bt = getWidget(node, "brush_type");
    if (bt) {
        body.appendChild(pillBar([
            { value: "paint", label: "Paint" },
            { value: "mask_only", label: "Mask only" },
        ], {
            value: bt.value || "paint",
            onChange: (v) => setWidgetValue(node, bt, v),
        }));
    }

    for (const w of (node.widgets || [])) {
        if (!w?.name?.startsWith(MASK_WIDGET_PREFIX)) continue;
        const o = readNumOpts(w, { min: 0, max: 1, step: 0.01 });
        const isInt = w.type === "number" && Number.isInteger(o.min) && Number.isInteger(o.max);
        body.appendChild(sliderRow(w.name.replace(/_/g, " "), {
            min: o.min, max: o.max, step: o.step,
            value: +w.value,
            format: !isInt && o.max <= 1 ? (v) => `${Math.round(v * 100)}%` : undefined,
            onChange: (v) => setWidgetValue(node, w, isInt ? (v | 0) : v),
        }));
    }

    wrap.appendChild(section("Output", body));
    return wrap;
}

function buildEditorCentre(node, editorCtrl, editorState) {
    const workspace = document.createElement("div");
    workspace.style.cssText = "width:100%;height:100%;position:relative;overflow:hidden;display:flex;align-items:center;justify-content:center;";

    const inner = document.createElement("div");
    inner.style.transformOrigin = "center center";
    inner.appendChild(editorCtrl.root);

    editorState.zoom = 1;
    editorState.fitZoom = 1;

    const applyZoom = () => {
        const z = editorState.zoom * editorState.fitZoom;
        inner.style.transform = `scale(${z})`;
        editorState._zoomLabel?.();
    };

    editorState.onFit = () => {
        const cw = workspace.clientWidth || 1;
        const ch = workspace.clientHeight || 1;
        const [pw, ph] = editorCtrl.size;
        editorState.fitZoom = Math.min(cw / pw, ch / ph) * 0.92;
        applyZoom();
    };

    const bar = zoomBar({
        onOut: () => { editorState.zoom = Math.max(0.1, editorState.zoom * 0.9); applyZoom(); },
        onIn: () => { editorState.zoom = Math.min(8, editorState.zoom * 1.1); applyZoom(); },
        onFit: () => { editorState.zoom = 1; editorState.onFit(); },
        getLabel: () => editorState._labelText?.() || "100%",
    });
    editorState._zoomLabel = () => {
        const lbl = bar.querySelector(".c2c-ui-zoombar__label");
        if (lbl) lbl.textContent = editorState._labelText();
    };
    editorState._labelText = () => `${Math.round(editorState.zoom * 100)}%`;

    bar.style.position = "absolute";
    bar.style.bottom = "12px";
    bar.style.left = "50%";
    bar.style.transform = "translateX(-50%)";
    bar.style.zIndex = "2";

    workspace.appendChild(inner);
    workspace.appendChild(bar);

    workspace.addEventListener("wheel", (e) => {
        if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            const factor = e.deltaY > 0 ? 0.9 : 1.1;
            editorState.zoom = Math.max(0.1, Math.min(8, editorState.zoom * factor));
            applyZoom();
        }
    }, { passive: false });

    const ro = new ResizeObserver(() => {
        if (editorState.zoom === 1) editorState.onFit();
    });
    ro.observe(workspace);
    editorState._disposeCentre = () => ro.disconnect();

    requestAnimationFrame(() => editorState.onFit());

    return workspace;
}

function openPaintEditor(node, nodeCtrl, stageApi) {
    const editorCtrl = new PaintCanvasController(node, { interactive: true, serialiseOnStroke: false });
    editorCtrl.copyFrom(nodeCtrl);

    const editorState = { tool: "brush" };
    const undoStack = createUndoStack(editorCtrl);
    undoStack.reset();

    let editorShell = null;
    let dirtyMarked = false;
    let saved = false;
    const widgetSnap = snapshotWidgetValues(node, editorWidgetNames(node));

    const syncUndoButtons = () => {
        const btns = editorShell?.el?.querySelectorAll(".c2c-ui-editor__header-actions button");
        if (!btns || btns.length < 2) return;
        btns[0].disabled = !undoStack.canUndo();
        btns[1].disabled = !undoStack.canRedo();
    };

    editorCtrl.onStrokeEnd = () => {
        undoStack.push();
        syncUndoButtons();
        if (!dirtyMarked) {
            dirtyMarked = true;
            editorShell?.setDirty(true);
        }
    };

    const recentRow = buildRecentColorRow(node, (hex) => {
        const colW = getWidget(node, "brush_color");
        if (colW) setWidgetValue(node, colW, hex);
    });

    const centre = buildEditorCentre(node, editorCtrl, editorState);
    const left = buildEditorLeft(node, editorCtrl, editorState, undoStack, recentRow);
    const right = buildEditorRight(node);

    const onUndo = () => { if (undoStack.undo()) syncUndoButtons(); };
    const onRedo = () => { if (undoStack.redo()) syncUndoButtons(); };

    node._mecPaintEditorOpen = true;

    const onKeyTool = (e) => {
        const t = e.target;
        if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName || ""))) return;
        if (e.key === "b" || e.key === "B") {
            editorState.tool = "brush";
            editorCtrl.setEraser(false);
            e.preventDefault();
        } else if (e.key === "e" || e.key === "E") {
            editorState.tool = "eraser";
            editorCtrl.setEraser(true);
            e.preventDefault();
        } else if (e.key === "[") {
            const w = getWidget(node, "brush_size");
            if (w) {
                const o = readNumOpts(w, { min: 1, max: 500, step: 1 });
                setWidgetValue(node, w, Math.max(o.min, (w.value | 0) - 2));
                editorCtrl._drawCursor();
            }
            e.preventDefault();
        } else if (e.key === "]") {
            const w = getWidget(node, "brush_size");
            if (w) {
                const o = readNumOpts(w, { min: 1, max: 500, step: 1 });
                setWidgetValue(node, w, Math.min(o.max, (w.value | 0) + 2));
                editorCtrl._drawCursor();
            }
            e.preventDefault();
        }
    };

    editorShell = openEditor({
        title: "Paint",
        left,
        right,
        centre,
        hints: [
            "B brush", "E eraser", "[ ] brush size",
            "Right-drag erases", "Ctrl+wheel zooms",
        ],
        onUndo,
        onRedo,
        onSave: () => {
            saved = true;
            const [w, h] = canvasDims(node);
            nodeCtrl.setSize(w, h);
            nodeCtrl.copyFrom(editorCtrl);
            nodeCtrl._synced = true;
            nodeCtrl.serialise();
            refreshNodeStage(stageApi, nodeCtrl, node);
            markGraphDirty(node);
            return true;
        },
        onClose: () => {
            node._mecPaintEditorOpen = false;
            if (!saved) {
                restoreWidgetValues(node, widgetSnap);
                const [w, h] = canvasDims(node);
                nodeCtrl.setSize(w, h);
                refreshNodeStage(stageApi, nodeCtrl, node);
            }
            editorShell?.el?.removeEventListener("keydown", onKeyTool, true);
            editorState._disposeCentre?.();
            editorCtrl.dispose();
        },
    });

    editorShell.el.addEventListener("keydown", onKeyTool, true);
    syncUndoButtons();
}

function buildNodePanel(node, nodeCtrl) {
    const [initW, initH] = canvasDims(node);
    const panelRoot = document.createElement("div");
    panelRoot.style.display = "flex";
    panelRoot.style.flexDirection = "column";
    panelRoot.style.gap = "8px";
    panelRoot.style.width = "100%";

    const stageApi = stage({
        aspect: initW / initH,
        empty: {
            title: "Advanced Paint Canvas",
            hint: "Open the editor to start painting",
        },
    });
    stageApi.setFooter(`${initW} \u00D7 ${initH}`);

    const openBtn = button("Open paint editor", {
        primary: true,
        block: true,
        onClick: () => openPaintEditor(node, nodeCtrl, stageApi),
    });

    // Clearing from the node cannot be undone, so it asks once more, in place.
    let armedUntil = 0;
    let armTimer = null;
    const clearRow = actionRow([{
        label: "Clear",
        onClick: (e) => {
            const btn = e?.currentTarget;
            const label = btn?.querySelector("span:last-child") || btn;
            if (Date.now() > armedUntil) {
                armedUntil = Date.now() + 3000;
                if (label) label.textContent = "Click again to clear";
                clearTimeout(armTimer);
                armTimer = setTimeout(() => { if (label) label.textContent = "Clear"; }, 3000);
                return;
            }
            armedUntil = 0;
            clearTimeout(armTimer);
            if (label) label.textContent = "Clear";
            nodeCtrl.clear();
            nodeCtrl._synced = true;
            nodeCtrl.serialise();
            refreshNodeStage(stageApi, nodeCtrl, node);
            markGraphDirty(node);
        },
    }]);

    panelRoot.appendChild(openBtn);
    panelRoot.appendChild(clearRow);
    panelRoot.appendChild(stageApi.el);

    const PANEL_MIN = 300;
    mountPanel(node, "paint_panel", panelRoot, { minHeight: PANEL_MIN });

    return stageApi;
}

function wireCanvasSizeSync(node, nodeCtrl, stageApi) {
    const sync = () => {
        if (node._mecPaintEditorOpen) return;
        const [w, h] = canvasDims(node);
        nodeCtrl.setSize(w, h);
        refreshNodeStage(stageApi, nodeCtrl, node);
    };
    for (const name of ["canvas_width", "canvas_height"]) {
        const w = getWidget(node, name);
        if (!w) continue;
        const cb = w.callback;
        w.callback = function (v) {
            cb?.call(this, v);
            sync();
        };
    }
    setTimeout(sync, 0);
}

function setupCanvasDataWidget(node) {
    let dataW = getWidget(node, "canvas_data");
    if (!dataW) {
        dataW = node.addWidget("text", "canvas_data", "", () => {}, { multiline: false });
    }
    if (dataW && !dataW.options) dataW.options = { multiline: false };
    hideWidget(dataW);
    return dataW;
}

function hideBrushWidgets(node) {
    for (const name of BRUSH_WIDGETS_HIDE) {
        hideWidget(getWidget(node, name));
    }
}

function editorWidgetNames(node) {
    const names = ["brush_type", "brush_color", "brush_opacity", "brush_hardness", "brush_size",
        "canvas_width", "canvas_height"];
    for (const w of (node.widgets || [])) {
        if (w?.name?.startsWith(MASK_WIDGET_PREFIX)) names.push(w.name);
    }
    return names;
}

function snapshotWidgetValues(node, names) {
    const snap = {};
    for (const n of names) {
        const w = getWidget(node, n);
        if (w) snap[n] = w.value;
    }
    return snap;
}

function restoreWidgetValues(node, snap) {
    for (const [name, value] of Object.entries(snap)) {
        const w = getWidget(node, name);
        if (w) setWidgetValue(node, w, value);
    }
}

app.registerExtension({
    name: "mec.advanced_paint",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onCreated?.apply(this, arguments);

            const nodeCtrl = new PaintCanvasController(this, {
                interactive: false,
                serialiseOnStroke: false,
            });
            this._mecPaint = nodeCtrl;

            const dataW = setupCanvasDataWidget(this);
            hideBrushWidgets(this);

            const stageApi = buildNodePanel(this, nodeCtrl);
            wireCanvasSizeSync(this, nodeCtrl, stageApi);

            nodeCtrl.onLoaded = () => refreshNodeStage(stageApi, nodeCtrl, this);

            const restore = () => {
                const v = dataW.value;
                if (v && typeof v === "string" && v.startsWith("data:")) {
                    nodeCtrl.loadFromDataURL(v);      // sets _synced on decode
                } else {
                    nodeCtrl._synced = true;          // nothing saved: the blank canvas IS the truth
                    refreshNodeStage(stageApi, nodeCtrl, this);
                }
            };
            setTimeout(restore, 30);

            // Reference image hook reserved for a future preview endpoint.
        };

        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (o) {
            // Only once the controller holds the real picture. A saved
            // painting decodes asynchronously after load; serialising before
            // that would write a blank canvas over it.
            try { if (this._mecPaint?._synced) this._mecPaint.serialise(); } catch (e) { __c2cReport("mec_advanced_paint", e); }
            onSerialize?.apply(this, arguments);
        };
    },
});
