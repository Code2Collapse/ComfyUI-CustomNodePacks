/** Full-screen modal mask editor. */
import { C } from "./palette.js";
import { c2cAlert, c2cConfirm } from "../_c2c_dialog.js";
import { screenToImage } from "./coords.js";
import { IMEEditor } from "./editor.js";
import { FrameStrip } from "./strip.js";

const TOOL_LABELS = {
    brush: "B", eraser: "E", rect: "R", ellipse: "O", polygon: "P", lasso: "L", bucket: "G",
};
const VIEW_LABELS = ["Overlay", "Matte", "Image"];
// Hotkey letter -> tool, derived from the labels so the toolbar and the keyboard cannot disagree.
const KEY_TO_TOOL = Object.fromEntries(Object.entries(TOOL_LABELS).map(([tool, key]) => [key.toLowerCase(), tool]));

const HOTKEY_TEXT =
    "B brush · E eraser · R rect · O ellipse · P polygon · L lasso · G bucket\n" +
    "Alt subtract · [ ] size · , . prev/next frame · F fit · 1 100%\n" +
    "V view mode · Ctrl+Z undo · Ctrl+Y redo · Enter save · Esc cancel shape / close";

function _formControlFocused(e) {
    const t = e.target;
    if (!t) return false;
    const tag = t.tagName;
    if (tag === "INPUT") {
        if ((t.type || "").toLowerCase() === "range") return false;
        return true;
    }
    return tag === "SELECT" || tag === "TEXTAREA" || t.isContentEditable;
}

function _mkRange(id, label, min, max, step, val, fmt) {
    const wrap = document.createElement("label");
    wrap.style.cssText = `display:inline-flex;align-items:center;gap:4px;color:${C.sub};font-size:12px;`;
    wrap.htmlFor = id;
    const cap = document.createElement("span");
    cap.textContent = label;
    const valEl = document.createElement("span");
    valEl.style.minWidth = "36px";
    valEl.textContent = fmt(val);
    const inp = document.createElement("input");
    inp.type = "range";
    inp.id = id;
    inp.min = String(min);
    inp.max = String(max);
    inp.step = String(step);
    inp.value = String(val);
    inp.style.width = "88px";
    wrap.append(cap, inp, valEl);
    return { wrap, inp, valEl, fmt };
}

export function openModal(node, editorId, onSaved) {
    if (!node?.graph) return;
    let closed = false;
    const ed = new IMEEditor(node, editorId);
    const strip = new FrameStrip();
    const prevFocus = document.activeElement;
    const overlay = document.createElement("div");
    overlay.dataset.imeModal = "1";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", "Image mask editor");
    overlay.style.cssText = [
        "position:fixed;inset:0;z-index:var(--c2c-z-modal,10000);",
        `background:${C.bg};display:flex;flex-direction:column;`,
        "font:13px/1.4 system-ui,sans-serif;color:" + C.text,
    ].join("");
    overlay.__imeEditor = ed;

    const banner = document.createElement("div");
    banner.hidden = true;
    banner.style.cssText = `background:${C.warn};color:#000;padding:6px 12px;text-align:center;`;
    banner.textContent = "Editing at reduced resolution; mask is upscaled on save.";

    const toolbar = document.createElement("div");
    toolbar.style.cssText = `display:flex;gap:6px;padding:8px;background:${C.panel};border-bottom:1px solid ${C.border};flex-wrap:wrap;align-items:center;`;

    const status = document.createElement("span");
    status.style.cssText = `margin-left:auto;color:${C.sub};`;

    const canvas = document.createElement("canvas");
    canvas.dataset.imeCanvas = "1";
    // flex-basis 0 + min-height 0: a flex item's min-height defaults to its content height, and a canvas's
    // content height is its last drawn pixel size, so without these the canvas could not shrink when the frame
    // strip appeared and pushed the strip and Save/Cancel below the window.
    canvas.style.cssText = "flex:1 1 0;min-height:0;width:100%;touch-action:none;cursor:crosshair;";
    canvas.tabIndex = 0;
    ed.dom.canvas = canvas;

    const stripHost = document.createElement("div");
    stripHost.hidden = true;

    const frameBar = document.createElement("div");
    frameBar.style.cssText = `display:flex;gap:8px;padding:8px;background:${C.panel};border-top:1px solid ${C.border};align-items:center;`;

    const navBar = document.createElement("div");
    navBar.style.cssText = "display:flex;gap:8px;align-items:center;";
    const btnPrev = document.createElement("button");
    btnPrev.textContent = "◀";
    const btnNext = document.createElement("button");
    btnNext.textContent = "▶";
    const btnCopyPrev = document.createElement("button");
    btnCopyPrev.textContent = "Copy previous frame";
    navBar.append(btnPrev, btnNext, btnCopyPrev);

    const actionBar = document.createElement("div");
    actionBar.style.cssText = "display:flex;gap:8px;margin-left:auto;";
    const btnSave = document.createElement("button");
    btnSave.textContent = "💾 Save";
    const btnCancel = document.createElement("button");
    btnCancel.textContent = "Cancel";
    actionBar.append(btnSave, btnCancel);

    for (const b of [btnPrev, btnNext, btnCopyPrev, btnSave, btnCancel]) {
        b.style.cssText = `padding:4px 10px;border:1px solid ${C.border};background:${C.panel};color:${C.text};cursor:pointer;border-radius:4px;`;
    }

    frameBar.append(navBar, actionBar);
    overlay.append(banner, toolbar, canvas, stripHost, frameBar);
    document.body.appendChild(overlay);

    const toolList = ["brush", "eraser", "rect", "ellipse", "polygon", "lasso", "bucket"];
    const toolBtns = {};
    for (const t of toolList) {
        const b = document.createElement("button");
        b.textContent = `${TOOL_LABELS[t] || t} ${t}`;
        b.style.cssText = `padding:4px 8px;border:1px solid ${C.border};background:${C.panel};color:${C.text};cursor:pointer;border-radius:4px;`;
        b.onclick = () => { ed.setTool(t); highlightTool(); };
        toolBtns[t] = b;
        toolbar.appendChild(b);
    }

    const sizeCtl = _mkRange("ime-brush-size", "Size", 1, 512, 1, ed.brushSize, (v) => `${v | 0}px`);
    const hardCtl = _mkRange("ime-brush-hardness", "Hard", 0, 1, 0.01, ed.brushHardness, (v) => Number(v).toFixed(2));
    const opCtl = _mkRange("ime-brush-opacity", "Opac", 0, 1, 0.01, ed.brushOpacity, (v) => Number(v).toFixed(2));
    const tolCtl = _mkRange("ime-bucket-tolerance", "Tol", 0, 255, 1, ed.bucketTolerance, (v) => String(v | 0));
    for (const c of [sizeCtl, hardCtl, opCtl, tolCtl]) toolbar.appendChild(c.wrap);

    const btnView = document.createElement("button");
    btnView.id = "ime-view-mode";
    btnView.textContent = VIEW_LABELS[ed.viewMode];
    btnView.style.cssText = `padding:4px 8px;border:1px solid ${C.border};background:${C.panel};color:${C.text};cursor:pointer;border-radius:4px;`;

    const btnHelp = document.createElement("button");
    btnHelp.textContent = "?";
    btnHelp.title = "Keyboard shortcuts";
    btnHelp.style.cssText = btnView.style.cssText;

    toolbar.append(btnView, btnHelp, status);

    function syncBrushSize(v) {
        const n = Math.max(1, Math.min(512, v | 0));
        ed.brushSize = n;
        sizeCtl.inp.value = String(n);
        sizeCtl.valEl.textContent = sizeCtl.fmt(n);
    }

    sizeCtl.inp.addEventListener("input", () => {
        syncBrushSize(Number(sizeCtl.inp.value));
    });
    hardCtl.inp.addEventListener("input", () => {
        ed.brushHardness = Number(hardCtl.inp.value);
        hardCtl.valEl.textContent = hardCtl.fmt(ed.brushHardness);
    });
    opCtl.inp.addEventListener("input", () => {
        ed.brushOpacity = Number(opCtl.inp.value);
        opCtl.valEl.textContent = opCtl.fmt(ed.brushOpacity);
    });
    tolCtl.inp.addEventListener("input", () => {
        ed.bucketTolerance = Number(tolCtl.inp.value) | 0;
        tolCtl.valEl.textContent = tolCtl.fmt(ed.bucketTolerance);
    });

    function relabelView() {
        btnView.textContent = VIEW_LABELS[ed.viewMode] || "Overlay";
    }

    function cycleViewUi() {
        ed.cycleView();
        relabelView();
    }

    btnView.onclick = () => { cycleViewUi(); canvas.focus(); };
    btnHelp.onclick = () => c2cAlert(HOTKEY_TEXT);

    function highlightTool() {
        for (const [k, b] of Object.entries(toolBtns)) {
            b.style.outline = k === ed.tool ? `2px solid ${C.accent}` : "";
        }
    }

    function updateStatus() {
        status.textContent =
            `frame ${ed.curFrame + 1}/${ed.frameCount} · ${ed.maskCount()} masks · ${ed.nativeW}×${ed.nativeH}`;
        navBar.style.display = ed.frameCount > 1 ? "flex" : "none";
        stripHost.hidden = ed.frameCount <= 1;
        strip.refresh();
    }

    function localXY(e) {
        const r = canvas.getBoundingClientRect();
        return screenToImage(e.clientX - r.left, e.clientY - r.top, ed.zoom, ed.panX, ed.panY);
    }

    function closeModal() {
        if (closed) return;
        closed = true;
        window.removeEventListener("keydown", onKey);
        window.removeEventListener("keyup", onKeyUp);
        try { canvasRo.disconnect(); } catch (_) { /* ignore */ }
        if (node.onRemoved === onRemovedWrapper) node.onRemoved = origRemoved;
        strip.dispose();
        ed.dispose();
        try { overlay.remove(); } catch (_) { /* node may be gone */ }
        try { if (prevFocus?.isConnected && typeof prevFocus.focus === "function") prevFocus.focus({ preventScroll: true }); } catch (_) { /* ignore */ }
    }

    async function tryClose(save) {
        if (save) {
            try {
                await ed.save();
                onSaved?.(ed.digest);
            } catch (e) {
                c2cAlert("Save failed:\n" + (e?.message || e));
                return;
            }
        } else if (ed.dirtyFrames.size) {
            const ok = await c2cConfirm("Discard unsaved mask changes?");
            if (!ok) return;
        }
        closeModal();
    }

    function endPointer(e) {
        if (!node.graph) { closeModal(); return; }
        const p = localXY(e);
        ed.finishPointerStroke(p.x, p.y, e);
        if (ed._baseBrush != null) ed.brushSize = ed._baseBrush;
    }

    canvas.addEventListener("pointerdown", (e) => {
        if (!node.graph) { closeModal(); return; }
        if (e.button !== 0 && e.button !== 1) return;
        canvas.setPointerCapture(e.pointerId);
        ed.subtract = e.altKey;
        ed._baseBrush = ed.brushSize;
        if (e.pointerType === "pen" && e.pressure > 0) {
            ed.brushSize = Math.max(1, ed._baseBrush * e.pressure);
        }
        const p = localXY(e);
        ed.pointerDown(p.x, p.y, e);
        e.preventDefault();
    });
    canvas.addEventListener("pointermove", (e) => {
        if (!node.graph) return;
        const p = localXY(e);
        ed.pointerMove(p.x, p.y, e);
    });
    canvas.addEventListener("pointerup", (e) => {
        endPointer(e);
        try { canvas.releasePointerCapture(e.pointerId); } catch (_) { /* ignore */ }
    });
    canvas.addEventListener("pointercancel", (e) => {
        endPointer(e);
        try { canvas.releasePointerCapture(e.pointerId); } catch (_) { /* ignore */ }
    });
    canvas.addEventListener("lostpointercapture", (e) => {
        if (ed.painting || ed.strokeStart) endPointer(e);
    });
    canvas.addEventListener("wheel", (e) => {
        const r = canvas.getBoundingClientRect();
        ed.wheelZoom(e.clientX - r.left, e.clientY - r.top, e.deltaY);
        e.preventDefault();
    }, { passive: false });
    canvas.addEventListener("dblclick", () => {
        if (ed.tool === "polygon") ed.closePolygon();
    });

    btnPrev.onclick = () => ed._switchFrame(ed.curFrame - 1).then(() => updateStatus());
    btnNext.onclick = () => ed._switchFrame(ed.curFrame + 1).then(() => updateStatus());
    btnCopyPrev.onclick = () => { ed.copyPrevFrame().then(() => updateStatus()); };
    btnSave.onclick = () => tryClose(true);
    btnCancel.onclick = () => tryClose(false);

    function onKey(e) {
        if (!node.graph) { closeModal(); return; }
        if (_formControlFocused(e)) return;
        if (e.ctrlKey && e.key === "z" && !e.shiftKey) { e.preventDefault(); ed.undoOp(); return; }
        if ((e.ctrlKey && e.shiftKey && e.key === "Z") || (e.ctrlKey && e.key === "y")) {
            e.preventDefault(); ed.redoOp(); return;
        }
        if (e.key === "Enter" && ed.tool === "polygon") { e.preventDefault(); ed.closePolygon(); return; }
        if (e.key === "Enter") { e.preventDefault(); tryClose(true); return; }
        if (e.key === "Escape") {
            e.preventDefault();
            if (ed.shapeInProgress()) {
                ed.cancelShapeInProgress();
                return;
            }
            tryClose(false);
            return;
        }
        if (e.key === "f" || e.key === "F") { ed.fit(); return; }
        if (e.key === "1") { ed.zoom100(); return; }
        if (e.key === "v" || e.key === "V") { cycleViewUi(); return; }
        if (e.key === " ") { e.preventDefault(); ed.spacePan = true; return; }
        if (e.key === "?" || (e.shiftKey && e.key === "/")) {
            c2cAlert(HOTKEY_TEXT);
            return;
        }
        if (e.key === ",") { ed._switchFrame(ed.curFrame - 1).then(() => updateStatus()); return; }
        if (e.key === ".") { ed._switchFrame(ed.curFrame + 1).then(() => updateStatus()); return; }
        if (e.key === "[") { syncBrushSize(ed.brushSize - 2); return; }
        if (e.key === "]") { syncBrushSize(ed.brushSize + 2); return; }
        if (e.key === "Backspace" && ed.tool === "polygon") { ed.popPolyPoint(); return; }
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        const t = KEY_TO_TOOL[e.key.toLowerCase()];
        if (t) { e.preventDefault(); ed.setTool(t); highlightTool(); }
    }
    function onKeyUp(e) {
        if (e.key === " ") ed.spacePan = false;
    }
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onKeyUp);

    const origRemoved = node.onRemoved;
    function onRemovedWrapper(...a) {
        closeModal();
        return origRemoved?.apply(this, a);
    }
    node.onRemoved = onRemovedWrapper;

    const canvasRo = new ResizeObserver(() => ed.requestDraw());
    canvasRo.observe(canvas);

    ed.onChange = updateStatus;
    highlightTool();
    ed.loadFrames().then((res) => {
        if (!node.graph) { closeModal(); return; }
        if (!res.ok) {
            c2cAlert(res.message || "No image source found.");
            closeModal();
            return;
        }
        if (res.banner) banner.hidden = false;
        if (ed.frameSource?.note) {
            banner.hidden = false;
            banner.textContent = ed.frameSource.note;
        }
        strip.mount(stripHost, ed, updateStatus);
        updateStatus();
        ed.fit();
        ed.draw();
        canvas.focus();
    });
}

