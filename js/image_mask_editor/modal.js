/** Full-screen modal mask editor. */
import { C } from "./palette.js";
import { c2cAlert, c2cConfirm, c2cPrompt } from "../_c2c_dialog.js";
import { screenToImage } from "./coords.js";
import { IMEEditor } from "./editor.js";
import { FrameStrip } from "./strip.js";
import * as api from "./api.js";

const TOOL_LABELS = {
    brush: "B", eraser: "E", rect: "R", ellipse: "O", polygon: "P", lasso: "L", bucket: "G", colour: "C",
    sam: "S", refine: "M",
};
const VIEW_LABELS = ["Overlay", "Matte", "Rubylith", "Outline", "Image"];
// Hotkey letter -> tool, derived from the labels so the toolbar and the keyboard cannot disagree.
const KEY_TO_TOOL = Object.fromEntries(Object.entries(TOOL_LABELS).map(([tool, key]) => [key.toLowerCase(), tool]));

const HOTKEY_TEXT =
    "B brush · E eraser · R rect · O ellipse · P polygon · L lasso · G bucket · C colour\n" +
    "Add / Subtract / Intersect modes (toolbar); Alt swaps Add↔Subtract for area tools\n" +
    "Colour: click sample · Shift+click add sample · Enter/Apply commit · Esc clear preview\n" +
    "S SAM: click +/Alt− · drag box · Enter/Apply commit · Esc clear\n" +
    "M refine: paint edge band · release to matte · Esc clear band\n" +
    "Alt+brush subtract · [ ] size · , . prev/next frame · F fit · 1 100%\n" +
    "V cycles Overlay → Matte → Rubylith → Outline → Image · pixel grid from 800% zoom\n" +
    "Pen pressure toggles (brush toolbar); mouse strokes ignore them\n" +
    "Ctrl+Shift+C copy mask · Ctrl+Shift+V paste mask · Frame menu: apply all / clear / import / export\n" +
    "Layers ▾ panel · Ctrl+Shift+N add layer · Alt+[ / Alt+] layer below / above\n" +
    "Copy/paste/clear/apply-to-all edit the active layer; export shows the merged stack\n" +
    "Ctrl+Z / Ctrl+Y undo / redo (per frame, shared 128 MB budget) · Enter save · Esc cancel shape / close";

function _formControlFocused(e) {
    const t = e.target;
    if (!t) return false;
    const tag = t.tagName;
    if (tag === "INPUT") {
        const type = (t.type || "").toLowerCase();
        if (type === "range" || type === "checkbox") return false;   // not text entry: hotkeys keep working
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

    const toolList = ["brush", "eraser", "rect", "ellipse", "polygon", "lasso", "bucket", "colour", "sam", "refine"];
    const toolBtns = {};
    const btnStyle = `padding:4px 8px;border:1px solid ${C.border};background:${C.panel};color:${C.text};cursor:pointer;border-radius:4px;`;
    for (const t of toolList) {
        const b = document.createElement("button");
        b.textContent = `${TOOL_LABELS[t] || t} ${t}`;
        b.style.cssText = btnStyle;
        b.onclick = () => { ed.setTool(t); highlightTool(); syncToolbarVisibility(); };
        toolBtns[t] = b;
        toolbar.appendChild(b);
    }

    const modeGroup = document.createElement("div");
    modeGroup.style.cssText = "display:inline-flex;gap:4px;align-items:center;";
    const modeBtns = {};
    for (const [mode, label] of [["add", "Add"], ["subtract", "Subtract"], ["intersect", "Intersect"]]) {
        const b = document.createElement("button");
        b.id = `ime-mode-${mode}`;
        b.textContent = label;
        b.style.cssText = btnStyle;
        b.onclick = () => { ed.setSelectionMode(mode); highlightMode(); };
        modeBtns[mode] = b;
        modeGroup.appendChild(b);
    }
    toolbar.appendChild(modeGroup);

    const brushOpts = document.createElement("div");
    brushOpts.style.cssText = "display:inline-flex;gap:6px;align-items:center;";
    const sizeCtl = _mkRange("ime-brush-size", "Size", 1, 512, 1, ed.brushSize, (v) => `${v | 0}px`);
    const hardCtl = _mkRange("ime-brush-hardness", "Hard", 0, 1, 0.01, ed.brushHardness, (v) => Number(v).toFixed(2));
    const opCtl = _mkRange("ime-brush-opacity", "Opac", 0, 1, 0.01, ed.brushOpacity, (v) => Number(v).toFixed(2));
    const presSizeLbl = document.createElement("label");
    presSizeLbl.style.cssText = `display:inline-flex;align-items:center;gap:4px;color:${C.sub};font-size:12px;`;
    const presSizeInp = document.createElement("input");
    presSizeInp.type = "checkbox";
    presSizeInp.id = "ime-pressure-size";
    presSizeLbl.append(presSizeInp, document.createTextNode("P→size"));
    const presOpLbl = document.createElement("label");
    presOpLbl.style.cssText = presSizeLbl.style.cssText;
    const presOpInp = document.createElement("input");
    presOpInp.type = "checkbox";
    presOpInp.id = "ime-pressure-opacity";
    presOpLbl.append(presOpInp, document.createTextNode("P→opac"));
    brushOpts.append(sizeCtl.wrap, hardCtl.wrap, opCtl.wrap, presSizeLbl, presOpLbl);

    const bucketOpts = document.createElement("div");
    bucketOpts.style.cssText = "display:inline-flex;gap:6px;align-items:center;";
    const tolCtl = _mkRange("ime-bucket-tolerance", "Tol", 0, 255, 1, ed.bucketTolerance, (v) => String(v | 0));
    bucketOpts.appendChild(tolCtl.wrap);

    const colourOpts = document.createElement("div");
    colourOpts.style.cssText = `display:inline-flex;gap:6px;align-items:center;color:${C.sub};font-size:12px;`;
    const spaceLbl = document.createElement("label");
    spaceLbl.style.cssText = "display:inline-flex;align-items:center;gap:4px;";
    const spaceCap = document.createElement("span");
    spaceCap.textContent = "Space";
    const spaceSel = document.createElement("select");
    spaceSel.id = "ime-colour-space";
    for (const s of ["rgb", "hsv", "lab"]) {
        const o = document.createElement("option");
        o.value = s; o.textContent = s.toUpperCase();
        spaceSel.appendChild(o);
    }
    spaceLbl.append(spaceCap, spaceSel);
    const tolColCtl = _mkRange("ime-colour-tol", "Tol", 0, 255, 1, ed.colourTolerance, (v) => String(v | 0));
    const softCtl = _mkRange("ime-colour-soft", "Soft", 0, 128, 1, ed.colourSoftness, (v) => String(v | 0));
    const contigLbl = document.createElement("label");
    contigLbl.style.cssText = "display:inline-flex;align-items:center;gap:4px;";
    const contigInp = document.createElement("input");
    contigInp.type = "checkbox";
    contigInp.id = "ime-colour-contig";
    contigLbl.append(contigInp, document.createTextNode("Contig"));
    const btnColourApply = document.createElement("button");
    btnColourApply.id = "ime-colour-apply";
    btnColourApply.textContent = "Apply";
    btnColourApply.style.cssText = btnStyle;
    colourOpts.append(spaceLbl, tolColCtl.wrap, softCtl.wrap, contigLbl, btnColourApply);

    const samOpts = document.createElement("div");
    samOpts.style.cssText = `display:inline-flex;gap:6px;align-items:center;color:${C.sub};font-size:12px;`;
    const samModelLbl = document.createElement("label");
    samModelLbl.style.cssText = "display:inline-flex;align-items:center;gap:4px;";
    const samModelCap = document.createElement("span");
    samModelCap.textContent = "Model";
    const samModelSel = document.createElement("select");
    samModelSel.id = "ime-sam-model";
    samModelSel.style.maxWidth = "260px";      // long "(needs the sam2 package)" names otherwise wrap the toolbar
    samModelLbl.append(samModelCap, samModelSel);
    const btnSamApply = document.createElement("button");
    btnSamApply.id = "ime-sam-apply";
    btnSamApply.textContent = "Apply";
    btnSamApply.style.cssText = btnStyle;
    samOpts.append(samModelLbl, btnSamApply);

    toolbar.append(brushOpts, bucketOpts, colourOpts, samOpts);

    const importInp = document.createElement("input");
    importInp.type = "file";
    importInp.accept = "image/png";
    importInp.id = "ime-import-mask";
    importInp.hidden = true;
    overlay.appendChild(importInp);

    const btnFrameMenu = document.createElement("button");
    btnFrameMenu.id = "ime-frame-menu";
    btnFrameMenu.textContent = "Frame ▾";
    btnFrameMenu.style.cssText = btnStyle;
    const frameMenu = document.createElement("div");
    frameMenu.style.cssText = [
        "position:fixed;z-index:calc(var(--c2c-z-modal,10000) + 1);",
        `background:${C.panel};border:1px solid ${C.border};border-radius:4px;`,
        "padding:4px 0;flex-direction:column;min-width:168px;display:none;",
    ].join("");
    const menuBtnStyle = `padding:6px 12px;border:none;background:transparent;color:${C.text};text-align:left;cursor:pointer;width:100%;font:13px system-ui,sans-serif;`;
    function _menuItem(label, fn) {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = label;
        b.style.cssText = menuBtnStyle;
        b.onclick = () => {
            frameMenu.style.display = "none";
            fn();
            canvas.focus();
        };
        return b;
    }
    frameMenu.append(
        _menuItem("Copy mask", () => { ed.copyMask(); updateStatus(); }),
        _menuItem("Paste mask", () => { ed.pasteMask(false); updateStatus(); }),
        _menuItem("Apply to all frames", async () => {
            const ok = await c2cConfirm(
                `Apply this frame's mask to all ${ed.frameCount} frame${ed.frameCount === 1 ? "" : "s"}?`,
            );
            if (ok) await ed.applyToAllFrames();
            updateStatus();
        }),
        _menuItem("Clear frame", () => { ed.clearFrame(); updateStatus(); }),
        _menuItem("Export PNG", () => ed.exportMaskPng()),
        _menuItem("Import PNG", () => importInp.click()),
    );
    document.body.appendChild(frameMenu);
    frameMenu.addEventListener("click", (ev) => ev.stopPropagation());
    btnFrameMenu.onclick = (ev) => {
        hideLayersPanel();
        const r = btnFrameMenu.getBoundingClientRect();
        frameMenu.style.left = `${r.left}px`;
        frameMenu.style.top = `${r.bottom + 2}px`;
        frameMenu.style.display = frameMenu.style.display === "none" ? "flex" : "none";
        ev.stopPropagation();
    };
    function hideFrameMenu() {
        frameMenu.style.display = "none";
    }
    function onImportChange() {
        const f = importInp.files?.[0];
        if (f) ed.importMaskPng(f).then(() => updateStatus());
        importInp.value = "";
    }
    window.addEventListener("click", hideFrameMenu);
    importInp.addEventListener("change", onImportChange);

    const btnLayersMenu = document.createElement("button");
    btnLayersMenu.id = "ime-layers-menu";
    btnLayersMenu.textContent = "Layers ▾";
    btnLayersMenu.style.cssText = btnStyle;
    const layersPanel = document.createElement("div");
    layersPanel.style.cssText = [
        "position:fixed;z-index:calc(var(--c2c-z-modal,10000) + 1);",
        `background:${C.panel};border:1px solid ${C.border};border-radius:4px;`,
        "padding:8px;flex-direction:column;gap:6px;min-width:280px;max-height:360px;overflow:auto;display:none;",
    ].join("");
    const layersList = document.createElement("div");
    layersList.style.cssText = "display:flex;flex-direction:column;gap:4px;";
    const btnAddLayer = document.createElement("button");
    btnAddLayer.type = "button";
    btnAddLayer.textContent = "Add layer";
    btnAddLayer.style.cssText = menuBtnStyle;
    layersPanel.append(layersList, btnAddLayer);
    document.body.appendChild(layersPanel);
    layersPanel.addEventListener("click", (ev) => ev.stopPropagation());

    function hideLayersPanel() {
        layersPanel.style.display = "none";
    }

    function refreshLayersPanel() {
        layersList.replaceChildren();
        const man = ed.getLayerManifest();
        for (let i = 0; i < man.layers.length; i++) {
            const layer = man.layers[i];
            const row = document.createElement("div");
            row.style.cssText = [
                "display:grid;grid-template-columns:auto 1fr auto auto auto auto auto;gap:4px;align-items:center;",
                "padding:4px;border-radius:4px;",
                layer.id === man.active ? `outline:2px solid ${C.accent}` : "",
            ].join("");
            const visBtn = document.createElement("button");
            visBtn.type = "button";
            visBtn.textContent = layer.visible !== false ? "👁" : "○";
            visBtn.title = "Visibility";
            visBtn.style.cssText = "border:none;background:transparent;cursor:pointer;padding:2px 4px;";
            visBtn.onclick = () => {
                ed.setLayerVisible(layer.id, layer.visible === false);
                refreshLayersPanel();
                updateStatus();
            };
            const nameBtn = document.createElement("button");
            nameBtn.type = "button";
            nameBtn.textContent = layer.name || layer.id;
            nameBtn.style.cssText = `border:none;background:transparent;color:${C.text};text-align:left;cursor:pointer;padding:2px 4px;`;
            nameBtn.onclick = () => { ed.setActiveLayer(layer.id); refreshLayersPanel(); updateStatus(); };
            nameBtn.ondblclick = async (ev) => {
                ev.stopPropagation();
                const neu = await c2cPrompt("Layer name", layer.name || "");
                if (neu != null) {
                    ed.renameLayer(layer.id, neu);
                    refreshLayersPanel();
                }
                canvas.focus();
            };
            const modeSel = document.createElement("select");
            modeSel.style.cssText = `font:12px system-ui;background:${C.panel};color:${C.text};border:1px solid ${C.border};`;
            for (const m of ["add", "subtract", "intersect"]) {
                const o = document.createElement("option");
                o.value = m;
                o.textContent = m;
                modeSel.appendChild(o);
            }
            modeSel.value = layer.mode || "add";
            modeSel.onchange = () => {
                ed.setLayerMode(layer.id, modeSel.value);
                refreshLayersPanel();
                updateStatus();
            };
            const lockBtn = document.createElement("button");
            lockBtn.type = "button";
            lockBtn.textContent = layer.locked ? "🔒" : "🔓";
            lockBtn.title = "Lock";
            lockBtn.style.cssText = visBtn.style.cssText;
            lockBtn.onclick = () => {
                ed.setLayerLocked(layer.id, !layer.locked);
                refreshLayersPanel();
            };
            const upBtn = document.createElement("button");
            upBtn.type = "button";
            upBtn.textContent = "▲";
            upBtn.disabled = i === 0;
            upBtn.style.cssText = visBtn.style.cssText;
            upBtn.onclick = () => { ed.moveLayer(layer.id, -1); refreshLayersPanel(); updateStatus(); };
            const downBtn = document.createElement("button");
            downBtn.type = "button";
            downBtn.textContent = "▼";
            downBtn.disabled = i === man.layers.length - 1;
            downBtn.style.cssText = visBtn.style.cssText;
            downBtn.onclick = () => { ed.moveLayer(layer.id, 1); refreshLayersPanel(); updateStatus(); };
            const delBtn = document.createElement("button");
            delBtn.type = "button";
            delBtn.textContent = "✕";
            delBtn.title = "Delete layer";
            delBtn.style.cssText = visBtn.style.cssText;
            delBtn.onclick = async () => {
                const ok = await c2cConfirm(`Delete layer "${layer.name || layer.id}"?`);
                if (ok) {
                    await ed.deleteLayer(layer.id);
                    refreshLayersPanel();
                    updateStatus();
                }
                canvas.focus();
            };
            row.append(visBtn, nameBtn, modeSel, lockBtn, upBtn, downBtn, delBtn);
            layersList.appendChild(row);
        }
    }

    btnLayersMenu.onclick = (ev) => {
        hideFrameMenu();
        const r = btnLayersMenu.getBoundingClientRect();
        layersPanel.style.left = `${r.left}px`;
        layersPanel.style.top = `${r.bottom + 2}px`;
        refreshLayersPanel();
        layersPanel.style.display = layersPanel.style.display === "none" ? "flex" : "none";
        ev.stopPropagation();
    };
    btnAddLayer.onclick = () => {
        if (ed.addLayer()) {
            refreshLayersPanel();
            updateStatus();
        }
        canvas.focus();
    };
    window.addEventListener("click", hideLayersPanel);

    const btnView = document.createElement("button");
    btnView.id = "ime-view-mode";
    btnView.textContent = VIEW_LABELS[ed.viewMode];
    btnView.style.cssText = `padding:4px 8px;border:1px solid ${C.border};background:${C.panel};color:${C.text};cursor:pointer;border-radius:4px;`;

    const btnHelp = document.createElement("button");
    btnHelp.textContent = "?";
    btnHelp.title = "Keyboard shortcuts";
    btnHelp.style.cssText = btnView.style.cssText;

    toolbar.append(btnFrameMenu, btnLayersMenu, btnView, btnHelp, status);

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
    presSizeInp.addEventListener("change", () => { ed.pressureSize = presSizeInp.checked; });
    presOpInp.addEventListener("change", () => { ed.pressureOpacity = presOpInp.checked; });
    tolCtl.inp.addEventListener("input", () => {
        ed.bucketTolerance = Number(tolCtl.inp.value) | 0;
        tolCtl.valEl.textContent = tolCtl.fmt(ed.bucketTolerance);
    });
    spaceSel.addEventListener("change", () => {
        ed.colourSpace = spaceSel.value;
        ed._rebuildColourPreview();
    });
    tolColCtl.inp.addEventListener("input", () => {
        ed.colourTolerance = Number(tolColCtl.inp.value) | 0;
        tolColCtl.valEl.textContent = tolColCtl.fmt(ed.colourTolerance);
        ed._rebuildColourPreview();
    });
    softCtl.inp.addEventListener("input", () => {
        ed.colourSoftness = Number(softCtl.inp.value) | 0;
        softCtl.valEl.textContent = softCtl.fmt(ed.colourSoftness);
        ed._rebuildColourPreview();
    });
    contigInp.addEventListener("change", () => {
        ed.colourContiguous = contigInp.checked;
        ed._rebuildColourPreview();
    });
    btnColourApply.onclick = () => { ed.applyColourPreview(); canvas.focus(); };
    btnSamApply.onclick = () => { ed.applySamPreview(); canvas.focus(); };
    samModelSel.addEventListener("change", () => {
        ed.samModel = samModelSel.value;
        ed._requestSam();
    });

    function highlightMode() {
        for (const [k, b] of Object.entries(modeBtns)) {
            b.style.outline = k === ed.selectionMode ? `2px solid ${C.accent}` : "";
        }
    }

    function syncToolbarVisibility() {
        const t = ed.tool;
        // style.display, not [hidden]: these groups carry an inline display, which beats the [hidden] rule
        const show = (el, on) => { el.style.display = on ? "inline-flex" : "none"; };
        show(brushOpts, t === "brush" || t === "eraser" || t === "refine");
        show(bucketOpts, t === "bucket");
        show(colourOpts, t === "colour");
        show(samOpts, t === "sam");
    }

    let refineAlertShown = false;
    ed.onStatus = (text, kind) => {
        updateStatus();
        if (kind === "need_weights" && !refineAlertShown) {
            refineAlertShown = true;
            c2cAlert(
                text || "ViTMatte weights are not installed.\n"
                + "Place a HuggingFace model folder under ComfyUI/models/vitmatte/.",
            );
        }
    };

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
        syncToolbarVisibility();
    }

    function updateStatus() {
        const base =
            `frame ${ed.curFrame + 1}/${ed.frameCount} · ${ed.maskCount()} masks · ${ed.nativeW}×${ed.nativeH}`;
        const extra = ed.getToolStatusLine();
        status.textContent = extra ? `${base} · ${extra}` : base;
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
        window.removeEventListener("keydown", onKeyCapture, true);
        window.removeEventListener("keyup", onKeyUp);
        window.removeEventListener("click", hideFrameMenu);
        window.removeEventListener("click", hideLayersPanel);
        importInp.removeEventListener("change", onImportChange);
        btnFrameMenu.onclick = null;
        btnLayersMenu.onclick = null;
        try { frameMenu.remove(); } catch (_) { /* ignore */ }
        try { layersPanel.remove(); } catch (_) { /* ignore */ }
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
    }

    canvas.addEventListener("pointerdown", (e) => {
        if (!node.graph) { closeModal(); return; }
        if (e.button !== 0 && e.button !== 1) return;
        canvas.setPointerCapture(e.pointerId);
        ed.subtract = e.altKey;
        const p = localXY(e);
        ed.pointerDown(p.x, p.y, e);
        e.preventDefault();
    });
    canvas.addEventListener("pointermove", (e) => {
        if (!node.graph) return;
        const p = localXY(e);
        ed.pointerMove(p.x, p.y, e);
    });
    canvas.addEventListener("pointerleave", () => { ed.hover = null; ed.requestDraw(); });
    canvas.addEventListener("pointerup", (e) => {
        endPointer(e);
        try { canvas.releasePointerCapture(e.pointerId); } catch (_) { /* ignore */ }
    });
    canvas.addEventListener("pointercancel", (e) => {
        endPointer(e);
        try { canvas.releasePointerCapture(e.pointerId); } catch (_) { /* ignore */ }
    });
    canvas.addEventListener("lostpointercapture", (e) => {
        if (ed.painting || ed.strokeStart || (ed.tool === "refine" && ed._refineStrokeStart)) endPointer(e);
    });
    canvas.addEventListener("wheel", (e) => {
        const r = canvas.getBoundingClientRect();
        ed.wheelZoom(e.clientX - r.left, e.clientY - r.top, e.deltaY);
        e.preventDefault();
    }, { passive: false });
    canvas.addEventListener("dblclick", () => {
        if (ed.tool === "polygon") ed.closePolygon();
    });
    ed.onToolChange = syncToolbarVisibility;

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
        if (e.key === "Enter" && ed.candidatePreviewActive()) {
            e.preventDefault();
            if (ed.colourPreviewActive()) ed.applyColourPreview();
            else ed.applySamPreview();
            return;
        }
        if (e.key === "Enter" && ed.tool === "polygon") { e.preventDefault(); ed.closePolygon(e); return; }
        if (e.key === "Enter") { e.preventDefault(); tryClose(true); return; }
        if (e.key === "Escape") {
            e.preventDefault();
            if (layersPanel.style.display !== "none") {
                hideLayersPanel();
                return;
            }
            if (frameMenu.style.display !== "none") {
                hideFrameMenu();
                return;
            }
            if (ed.shapeInProgress() && !ed.candidatePreviewActive()) {
                ed.cancelShapeInProgress();
                return;
            }
            if (ed.tool === "sam" && (ed.samPoints.length > 0 || ed.samBox || ed.candidatePreview)) {
                ed.clearSamState();
                return;
            }
            if (ed.tool === "refine") {
                ed.clearRefineState();
                return;
            }
            if (ed.colourPreviewActive()) {
                ed.clearColourPreview();
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
        if (e.ctrlKey && e.shiftKey && (e.key === "c" || e.key === "C")) {
            e.preventDefault(); ed.copyMask(); updateStatus(); return;
        }
        if (e.ctrlKey && e.shiftKey && (e.key === "v" || e.key === "V")) {
            e.preventDefault(); ed.pasteMask(e.altKey); updateStatus(); return;
        }
        if (e.ctrlKey && e.shiftKey && (e.key === "n" || e.key === "N")) {
            e.preventDefault();
            if (ed.addLayer()) {
                if (layersPanel.style.display !== "none") refreshLayersPanel();
                updateStatus();
            }
            return;
        }
        if (e.altKey && e.key === "[") {
            e.preventDefault(); ed.selectLayerBelow(); updateStatus(); return;
        }
        if (e.altKey && e.key === "]") {
            e.preventDefault(); ed.selectLayerAbove(); updateStatus(); return;
        }
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        const t = KEY_TO_TOOL[e.key.toLowerCase()];
        if (t) { e.preventDefault(); ed.setTool(t); highlightTool(); syncToolbarVisibility(); }
    }
    function onKeyUp(e) {
        if (e.key === " ") ed.spacePan = false;
    }
    // Capture phase on window, so the editor sees keys before ComfyUI's window-level keybindings; a key it handled
    // goes no further. As a bubble listener added after the core one, Ctrl+Z undid the stroke AND reloaded the
    // whole graph behind the modal (ORDERS A9, L2.15).
    function onKeyCapture(e) { onKey(e); if (e.defaultPrevented) e.stopPropagation(); }
    window.addEventListener("keydown", onKeyCapture, true);
    window.addEventListener("keyup", onKeyUp);

    const origRemoved = node.onRemoved;
    function onRemovedWrapper(...a) {
        closeModal();
        return origRemoved?.apply(this, a);
    }
    node.onRemoved = onRemovedWrapper;

    // The toolbar re-wraps when the tool options change and the frame strip/banner come and go, which moves
    // the canvas's top edge; shift the pan by the same amount so the image stays still on screen.
    // ed.viewTop is the canvas top the current pan was set for (fit and 100% re-anchor it).
    const canvasRo = new ResizeObserver(() => {
        const top = canvas.getBoundingClientRect().top;
        if (ed.viewTop != null && top !== ed.viewTop) ed.panY += ed.viewTop - top;
        ed.viewTop = top;
        ed.requestDraw();
    });
    canvasRo.observe(canvas);

    ed.onChange = updateStatus;
    ed.onLayersChange = () => {
        if (layersPanel.style.display !== "none") refreshLayersPanel();
    };
    highlightTool();
    highlightMode();
    syncToolbarVisibility();
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
        api.samModels().then((m) => {
            const unavailable = m.unavailable || {};
            const allUnavailable = (m.models || []).length > 0
                && (m.models || []).every((name) => unavailable[name]);
            samModelSel.replaceChildren();
            if (!m.default) {           // nothing installed: no silent multi-GB download on the first click
                const o = document.createElement("option");
                o.value = "";
                if (allUnavailable) {
                    const reason = unavailable[m.models[0]] || "";
                    o.textContent = "SAM unavailable - see tooltip";
                    o.title = reason;
                } else {
                    o.textContent = "choose a model";
                }
                samModelSel.appendChild(o);
            }
            for (const name of m.models || []) {
                const o = document.createElement("option");
                o.value = name;
                const reason = unavailable[name];
                if (reason) {
                    o.disabled = true;
                    o.textContent = name + " (needs the sam2 package)";
                    o.title = reason;
                } else {
                    o.textContent = name;
                }
                samModelSel.appendChild(o);
            }
            // An option's title only shows while the list is open: the select carries the reason too, and the
            // editor quotes it when SAM is clicked without a model.
            const why = allUnavailable ? (unavailable[m.models[0]] || "") : "";
            samModelSel.title = why;
            ed.samUnavailable = why;
            ed.samModel = m.default || "";
            samModelSel.value = ed.samModel;
        }).catch((e) => console.warn("[IME] sam models:", e));
        strip.mount(stripHost, ed, updateStatus);
        updateStatus();
        ed.fit();
        ed.draw();
        canvas.focus();
    });
}

