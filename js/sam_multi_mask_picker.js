import { app } from "../../scripts/app.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";
import { ensureC2CKit } from "./_c2c_ui_kit.js";

/**
 * MEC – SAM Multi-Mask Picker Widget  (REAL DOM — rewritten 2026-07-24)
 *
 * Shows the 3 SAM candidate masks as genuine clickable <div> tiles: a real
 * <img> source backdrop with the mask PNG composited over it in CSS, a real
 * score pill + confidence bar, real CSS hover/selected/focus states.
 *
 * REPLACES: a version that painted the entire grid onto the node canvas
 * (22 ctx.* calls vs 1 DOM element) and hand-rolled hit-rectangles to work
 * out which tile you clicked. A grid of clickable thumbnails is the textbook
 * DOM case — canvas-painting it meant blurry text at non-100% zoom, hit-rects
 * that drift from what was drawn, no hover transitions, no keyboard focus,
 * and a full repaint per frame. The mask/image PIXELS still live in <img>
 * tags (browser-decoded, GPU-composited); nothing is redrawn per frame now.
 *
 * The mask PNGs are grayscale L-mode (pixel value == mask probability), so
 * the red overlay is done with `mix-blend-mode:screen` + a hue filter rather
 * than the old per-tile offscreen-canvas composite pass.
 *
 * Contracts preserved exactly (verify against the Python node before changing):
 *   - `selected_index` INT widget is the source of truth for the choice
 *   - keyboard 1/2/3 quick-picks mask 0/1/2
 *   - execution payload: `message.scores` (JSON array; legacy output.scores /
 *     output.ui.scores paths still accepted) and `message.mask_thumbs`
 *     ([{filename, subfolder, type}, ...])
 *   - source image read from the connected IMAGE input node's imgs[0].src
 */

const TARGET = "SamMultiMaskPickerMEC";
const N_MASKS = 3;

function ensurePickerStyles() {
    if (document.getElementById("mec-mask-picker-styles")) return;
    const el = document.createElement("style");
    el.id = "mec-mask-picker-styles";
    el.textContent = `
.mec-mp-root{display:flex;flex-direction:column;gap:6px;width:100%;height:100%;
  box-sizing:border-box;padding:6px;min-height:0;font:11px ui-sans-serif,system-ui;}
.mec-mp-head{display:flex;align-items:center;gap:6px;flex:0 0 auto;color:#8b93a7;}
.mec-mp-head b{color:#e6e9f0;font-weight:600;}
.mec-mp-grid{display:flex;gap:6px;flex:1 1 auto;min-height:0;}
.mec-mp-tile{position:relative;flex:1 1 0;min-width:0;border-radius:6px;overflow:hidden;
  border:1px solid #3a3a3a;background:#1e1e24;cursor:pointer;box-sizing:border-box;
  transition:border-color .12s ease,box-shadow .12s ease,transform .08s ease;
  display:flex;align-items:center;justify-content:center;}
.mec-mp-tile:hover{border-color:#5a5a5a;transform:translateY(-1px);}
.mec-mp-tile.sel{border-color:#5b9dd9;border-width:2px;box-shadow:0 0 0 1px rgba(91,157,217,.4);}
.mec-mp-tile:focus-visible{outline:2px solid #89b4fa;outline-offset:1px;}
.mec-mp-src,.mec-mp-mask{position:absolute;inset:0;width:100%;height:100%;
  object-fit:contain;pointer-events:none;user-select:none;-webkit-user-drag:none;}
.mec-mp-mask{mix-blend-mode:screen;opacity:.55;filter:sepia(1) saturate(6) hue-rotate(-40deg);}
.mec-mp-empty{color:#6b7280;font-size:10px;text-align:center;padding:6px;
  pointer-events:none;white-space:pre-line;line-height:1.4;}
.mec-mp-badge{position:absolute;top:4px;left:4px;min-width:15px;height:15px;border-radius:3px;
  background:rgba(0,0,0,.66);color:#e6e9f0;font-size:9px;line-height:15px;text-align:center;
  padding:0 3px;pointer-events:none;}
.mec-mp-tile.sel .mec-mp-badge{background:#5b9dd9;color:#08111c;font-weight:700;}
.mec-mp-score{position:absolute;left:0;right:0;bottom:0;padding:2px 5px;font-size:10px;
  color:#e6e9f0;background:rgba(0,0,0,.62);pointer-events:none;
  display:flex;justify-content:space-between;gap:4px;}
.mec-mp-score .v{font-variant-numeric:tabular-nums;color:#a6e3a1;}
.mec-mp-bar{position:absolute;left:0;bottom:0;height:2px;background:#a6e3a1;pointer-events:none;}
`;
    document.head.appendChild(el);
}

function installPicker(node) {
    if (node._mecPickerDom) return;
    ensureC2CKit();
    ensurePickerStyles();

    const state = {
        scores: [0, 0, 0],
        maskUrls: [null, null, null],
        srcUrl: null,
        lastSrcUrl: null,
    };

    const root = document.createElement("div");
    root.className = "mec-mp-root";

    const head = document.createElement("div");
    head.className = "mec-mp-head";
    const headLabel = document.createElement("b");
    headLabel.textContent = "SAM candidates";
    const headHint = document.createElement("span");
    headHint.textContent = "click a mask · or press 1 / 2 / 3";
    head.append(headLabel, headHint);
    root.appendChild(head);

    const grid = document.createElement("div");
    grid.className = "mec-mp-grid";
    grid.setAttribute("role", "radiogroup");
    grid.setAttribute("aria-label", "SAM mask candidates");
    root.appendChild(grid);

    const tiles = [];
    for (let i = 0; i < N_MASKS; i++) {
        const tile = document.createElement("div");
        tile.className = "mec-mp-tile";
        tile.tabIndex = 0;
        tile.setAttribute("role", "radio");
        tile.setAttribute("aria-label", `Mask candidate ${i + 1}`);

        const srcImg = document.createElement("img");
        srcImg.className = "mec-mp-src";
        srcImg.draggable = false;
        srcImg.style.display = "none";

        const maskImg = document.createElement("img");
        maskImg.className = "mec-mp-mask";
        maskImg.draggable = false;
        maskImg.style.display = "none";

        const empty = document.createElement("div");
        empty.className = "mec-mp-empty";
        empty.textContent = "queue to\ngenerate";

        const badge = document.createElement("div");
        badge.className = "mec-mp-badge";
        badge.textContent = String(i + 1);

        const score = document.createElement("div");
        score.className = "mec-mp-score";
        const sLabel = document.createElement("span");
        sLabel.textContent = "score";
        const sVal = document.createElement("span");
        sVal.className = "v";
        sVal.textContent = "—";
        score.append(sLabel, sVal);

        const bar = document.createElement("div");
        bar.className = "mec-mp-bar";
        bar.style.width = "0%";

        tile.append(srcImg, maskImg, empty, badge, score, bar);
        grid.appendChild(tile);

        const pick = (e) => {
            e?.preventDefault?.();
            e?.stopPropagation?.();
            setSelected(i);
        };
        tile.addEventListener("pointerdown", pick);
        tile.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") pick(e);
        });

        tiles.push({ tile, srcImg, maskImg, empty, sVal, bar });
    }

    function currentIndex() {
        const w = node.widgets?.find((x) => x.name === "selected_index");
        return Math.max(0, Math.min(N_MASKS - 1, Number(w?.value) || 0));
    }

    function setSelected(idx) {
        idx = Math.max(0, Math.min(N_MASKS - 1, idx));
        const w = node.widgets?.find((x) => x.name === "selected_index");
        if (w) {
            w.value = idx;
            try { w.callback?.(idx); }
            catch (err) { __c2cReport?.("sam_multi_mask_picker.callback", err); }
        }
        syncSelection();
        node.graph?.setDirtyCanvas(true, true);
    }

    function syncSelection() {
        const sel = currentIndex();
        tiles.forEach((t, i) => {
            t.tile.classList.toggle("sel", i === sel);
            t.tile.setAttribute("aria-checked", String(i === sel));
        });
    }

    function renderTiles() {
        for (let i = 0; i < N_MASKS; i++) {
            const t = tiles[i];
            const mUrl = state.maskUrls[i];
            const sUrl = state.srcUrl;
            if (sUrl && t.srcImg.getAttribute("src") !== sUrl) t.srcImg.src = sUrl;
            t.srcImg.style.display = sUrl ? "block" : "none";
            if (mUrl && t.maskImg.getAttribute("src") !== mUrl) t.maskImg.src = mUrl;
            t.maskImg.style.display = mUrl ? "block" : "none";
            t.empty.style.display = (sUrl || mUrl) ? "none" : "block";

            const sc = Number(state.scores[i]);
            const has = Number.isFinite(sc) && sc > 0;
            t.sVal.textContent = has ? sc.toFixed(3) : "—";
            t.bar.style.width = has ? `${Math.max(0, Math.min(1, sc)) * 100}%` : "0%";
        }
        syncSelection();
    }

    // Source-image discovery — unchanged contract: the upstream node's own
    // preview image (subgraph-safe link resolution via the owning graph).
    function tryLoadSourceImage() {
        try {
            const imgInput = node.inputs?.[0];
            if (!imgInput?.link) return;
            const ownerGraph = node.graph || app.graph;
            const linkInfo = ownerGraph?.links?.[imgInput.link];
            if (!linkInfo) return;
            const sourceNode = ownerGraph.getNodeById?.(linkInfo.origin_id);
            const url = sourceNode?.imgs?.[0]?.src;
            if (url && url !== state.lastSrcUrl) {
                state.lastSrcUrl = url;
                state.srcUrl = url;
                renderTiles();
            }
        } catch (err) {
            __c2cReport?.("sam_multi_mask_picker.source", err);
        }
    }

    function receiveMaskThumbnails(thumbInfos) {
        if (!Array.isArray(thumbInfos)) return;
        for (let i = 0; i < N_MASKS; i++) {
            const info = thumbInfos[i];
            if (!info?.filename) { state.maskUrls[i] = null; continue; }
            state.maskUrls[i] =
                `/view?filename=${encodeURIComponent(info.filename)}` +
                `&subfolder=${encodeURIComponent(info.subfolder || "")}` +
                `&type=${encodeURIComponent(info.type || "temp")}` +
                `&t=${Date.now()}`;
        }
        renderTiles();
    }

    node.addDOMWidget("mask_picker_display", "mask_picker", root, {
        getValue: () => "",
        setValue: () => {},
        serialize: false,
    });
    if (!node.size || node.size[1] < 260) {
        node.size = [Math.max(node.size?.[0] || 0, 320), Math.max(node.size?.[1] || 0, 260)];
    }

    const poll = setInterval(tryLoadSourceImage, 1500);
    const origRemoved = node.onRemoved;
    node.onRemoved = function () {
        origRemoved?.apply(this, arguments);
        clearInterval(poll);
    };

    node._mecPickerDom = {
        renderTiles, syncSelection, receiveMaskThumbnails, tryLoadSourceImage,
        setScores: (s) => { state.scores = s; renderTiles(); },
        setSelected,
    };

    tryLoadSourceImage();
    renderTiles();
}

if (!(app.extensions || []).some((e) => e?.name === "MEC.SamMultiMaskPicker")) {
    app.registerExtension({
        name: "MEC.SamMultiMaskPicker",

        async beforeRegisterNodeDef(nodeType, nodeData) {
            if (nodeData.name !== TARGET) return;

            // Keyboard 1/2/3 quick-pick (contract preserved).
            const onKeyDown = nodeType.prototype.onKeyDown;
            nodeType.prototype.onKeyDown = function (event) {
                if (onKeyDown) {
                    const handled = onKeyDown.apply(this, arguments);
                    if (handled) return true;
                }
                const k = event?.key;
                if (k >= "1" && k <= String(N_MASKS) && this._mecPickerDom) {
                    this._mecPickerDom.setSelected(parseInt(k, 10) - 1);
                    return true;
                }
                return false;
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                onConfigure?.apply(this, arguments);
                this._mecPickerDom?.syncSelection();
            };

            // Scores + real mask thumbnails arrive on execution.
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                if (!this._mecPickerDom || !message) return;
                try {
                    let parsed = null;
                    if (Array.isArray(message.scores) && message.scores.length > 0) {
                        parsed = JSON.parse(message.scores[0]);
                    } else if (message.output?.scores) {
                        const raw = message.output.scores;
                        parsed = JSON.parse(typeof raw === "string" ? raw : raw[0]);
                    } else if (message.output?.ui?.scores) {
                        parsed = JSON.parse(message.output.ui.scores);
                    }
                    if (Array.isArray(parsed) && parsed.length >= N_MASKS) {
                        this._mecPickerDom.setScores(parsed.slice(0, N_MASKS));
                    }
                } catch (err) { __c2cReport?.("sam_multi_mask_picker.scores", err); }

                if (Array.isArray(message.mask_thumbs) && message.mask_thumbs.length > 0) {
                    this._mecPickerDom.receiveMaskThumbnails(message.mask_thumbs);
                }
                this._mecPickerDom.tryLoadSourceImage();
            };
        },

        async nodeCreated(node) {
            if (node.comfyClass !== TARGET) return;
            try { installPicker(node); }
            catch (err) { __c2cReport?.("sam_multi_mask_picker.install", err, "sam_multi_mask_picker"); }
        },
    });
}
