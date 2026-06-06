/**
 * c2c_face_pose_delta.js — Frontend editor for FacePoseDeltaCoreMEC.
 *
 * Adds an "🖐 Edit landmark deltas…" button to every FacePoseDeltaCoreMEC
 * node. The button opens a modal that authors the `keyframe_edits_json`
 * widget without making the user type JSON by hand:
 *
 *   • Pick a frame (slider).
 *   • Pick a landmark index — either by typing it or by clicking one of
 *     the MediaPipe FaceMesh shortcut buttons (eyes, brows, mouth, nose).
 *   • Adjust dx / dy in anchor-relative units (1.0 = inter-ocular span).
 *   • Click "Add / replace keyframe" — the keyframe table updates and the
 *     live JSON preview rebuilds.
 *   • Repeat for as many keyframes as you like.
 *   • Pick `ease` and `extrapolate`.
 *   • "Save" writes the JSON back into the node widget.
 *
 * The math contract matches `nodes/face_pose_delta.py`: deltas are stored
 * keyed by landmark index per keyframe and propagated server-side using
 * eased blend weights. This editor is just the authoring surface.
 *
 * License: Apache-2.0
 */

import { app } from "../../scripts/app.js";
import { c2cConfirm } from "./_c2c_dialog.js";

// ─── MediaPipe FaceMesh landmark shortcuts ──────────────────────────────
// Curated subset — the indices a beginner would most plausibly want to
// edit. Pulled from the MediaPipe FaceMesh canonical_face_model.obj.
const MP_SHORTCUTS = [
    { label: "L eye outer",  idx: 33  },
    { label: "R eye outer",  idx: 263 },
    { label: "L eye inner",  idx: 133 },
    { label: "R eye inner",  idx: 362 },
    { label: "L brow inner", idx: 55  },
    { label: "R brow inner", idx: 285 },
    { label: "L brow outer", idx: 65  },
    { label: "R brow outer", idx: 295 },
    { label: "L mouth",      idx: 61  },
    { label: "R mouth",      idx: 291 },
    { label: "Upper lip",    idx: 13  },
    { label: "Lower lip",    idx: 14  },
    { label: "Nose tip",     idx: 1   },
    { label: "Chin",         idx: 152 },
];

const EASES = ["linear", "ease_in", "ease_out", "smooth_step"];
const EXTRAPS = ["hold", "zero", "loop"];

const STYLE_ID = "mec-fpd-style";
const MODAL_ID = "mec-fpd-modal";

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
#${MODAL_ID}-backdrop {
    position: fixed; inset: 0; background: rgba(0,0,0,0.55);
    z-index: var(--c2c-z-modal); display: none;
}
#${MODAL_ID}-backdrop.visible { display: block; }
#${MODAL_ID} {
    position: fixed; top: 50%; left: 50%; transform: translate(-50%,-50%);
    z-index: var(--c2c-z-modal); width: 640px; max-width: 95vw; max-height: 92vh;
    background: var(--c2c-bg); color: var(--c2c-fg); border: 1px solid var(--c2c-border);
    border-radius: 10px; box-shadow: 0 8px 32px rgba(0,0,0,0.6);
    font: 13px/1.45 system-ui, sans-serif; padding: 18px 20px;
    overflow-y: auto;
}
#${MODAL_ID} h3 { margin: 0 0 4px; color: var(--c2c-lavender); font-size: 16px; }
#${MODAL_ID} p.hint { margin: 0 0 14px; color: var(--c2c-overlay2); font-size: 11px; }
#${MODAL_ID} .row { display: flex; gap: 8px; align-items: center; margin: 6px 0; }
#${MODAL_ID} .row label { min-width: 90px; color: var(--c2c-sub); font-size: 12px; }
#${MODAL_ID} .row input[type="number"],
#${MODAL_ID} .row input[type="text"] {
    background: var(--c2c-bg2); color: var(--c2c-fg); border: 1px solid var(--c2c-surface1);
    border-radius: 4px; padding: 4px 6px; width: 90px; font: inherit;
}
#${MODAL_ID} .row input[type="range"] { flex: 1; }
#${MODAL_ID} select {
    background: var(--c2c-bg2); color: var(--c2c-fg); border: 1px solid var(--c2c-surface1);
    border-radius: 4px; padding: 4px 6px; font: inherit;
}
#${MODAL_ID} .shortcuts {
    display: flex; flex-wrap: wrap; gap: 4px; margin: 6px 0 12px;
}
#${MODAL_ID} .shortcuts button {
    background: var(--c2c-border); color: var(--c2c-fg); border: 1px solid var(--c2c-surface1);
    border-radius: 4px; padding: 3px 8px; font: 11px inherit; cursor: pointer;
}
#${MODAL_ID} .shortcuts button:hover { background: var(--c2c-surface1); }
#${MODAL_ID} .shortcuts button.active {
    background: var(--c2c-blue); color: var(--c2c-bg3); border-color: var(--c2c-blue);
}
#${MODAL_ID} .kf-table {
    width: 100%; margin: 10px 0; border-collapse: collapse;
    font-size: 12px;
}
#${MODAL_ID} .kf-table th, #${MODAL_ID} .kf-table td {
    border-bottom: 1px solid var(--c2c-border); padding: 4px 6px; text-align: left;
}
#${MODAL_ID} .kf-table th { color: var(--c2c-sub); font-weight: 600; }
#${MODAL_ID} .kf-table .del {
    background: transparent; color: var(--c2c-red); border: 1px solid var(--c2c-red);
    border-radius: 3px; padding: 1px 6px; cursor: pointer; font: 11px inherit;
}
#${MODAL_ID} .kf-table .del:hover { background: var(--c2c-red); color: var(--c2c-bg3); }
#${MODAL_ID} textarea {
    width: 100%; min-height: 100px; box-sizing: border-box;
    background: var(--c2c-bg2); color: var(--c2c-green); border: 1px solid var(--c2c-surface1);
    border-radius: 4px; padding: 6px; font: 11px ui-monospace, monospace;
    resize: vertical;
}
#${MODAL_ID} .btn-row { display: flex; gap: 8px; margin-top: 14px; justify-content: flex-end; }
#${MODAL_ID} .btn {
    background: var(--c2c-surface1); color: var(--c2c-fg); border: 1px solid var(--c2c-surface2);
    border-radius: 5px; padding: 6px 14px; cursor: pointer; font: inherit;
}
#${MODAL_ID} .btn:hover { background: var(--c2c-surface2); }
#${MODAL_ID} .btn.primary { background: var(--c2c-blue); color: var(--c2c-bg3); border-color: var(--c2c-blue); }
#${MODAL_ID} .btn.primary:hover { background: var(--c2c-lavender); }
#${MODAL_ID} .btn.add { background: var(--c2c-green); color: var(--c2c-bg3); border-color: var(--c2c-green); }
#${MODAL_ID} .btn.add:hover { background: var(--c2c-teal); }
#${MODAL_ID} .err { color: var(--c2c-red); font-size: 11px; margin: 4px 0; }
    `.trim();
    document.head.appendChild(s);
}

// ─── Modal state ────────────────────────────────────────────────────────
// Single shared modal — we keep it simple and recreate the data each open.
function _openEditor(node, widget) {
    _injectStyle();

    // Parse current widget value (defensive)
    let state = { keyframes: [], ease: "smooth_step", extrapolate: "hold" };
    try {
        const parsed = JSON.parse(widget.value || "{}");
        if (parsed && typeof parsed === "object") {
            if (Array.isArray(parsed.keyframes)) state.keyframes = parsed.keyframes;
            if (typeof parsed.ease === "string") state.ease = parsed.ease;
            if (typeof parsed.extrapolate === "string") state.extrapolate = parsed.extrapolate;
        }
    } catch { /* ignore — treat as empty */ }

    // Working buffer the editor mutates
    let workingKfs = JSON.parse(JSON.stringify(state.keyframes));

    // ── Build DOM ──
    let backdrop = document.getElementById(`${MODAL_ID}-backdrop`);
    if (backdrop) backdrop.remove();
    backdrop = document.createElement("div");
    backdrop.id = `${MODAL_ID}-backdrop`;
    backdrop.classList.add("visible");

    const modal = document.createElement("div");
    modal.id = MODAL_ID;
    modal.innerHTML = `
        <h3>🖐 Face / Pose Delta Editor</h3>
        <p class="hint">Anchor-relative units. <b>1.0</b> = inter-ocular distance.
        Typical edit magnitude: <b>0.02 – 0.10</b>. Edits follow the face as the head moves.</p>

        <div class="row">
            <label>Frame</label>
            <input type="number" id="fpd-frame" value="0" min="0" max="10000" step="1">
            <input type="range" id="fpd-frame-slider" value="0" min="0" max="120" step="1">
        </div>

        <div class="row">
            <label>Landmark idx</label>
            <input type="number" id="fpd-idx" value="61" min="0" max="100000" step="1">
            <span style="color:var(--c2c-overlay2);font-size:11px" id="fpd-idx-name"></span>
        </div>

        <div class="shortcuts" id="fpd-shortcuts"></div>

        <div class="row">
            <label>dx (rel)</label>
            <input type="number" id="fpd-dx" value="0" step="0.01">
            <input type="range" id="fpd-dx-slider" min="-0.5" max="0.5" step="0.005" value="0">
        </div>
        <div class="row">
            <label>dy (rel)</label>
            <input type="number" id="fpd-dy" value="0" step="0.01">
            <input type="range" id="fpd-dy-slider" min="-0.5" max="0.5" step="0.005" value="0">
        </div>

        <div class="row">
            <button class="btn add" id="fpd-add">＋ Add / replace keyframe</button>
            <button class="btn" id="fpd-clear" style="margin-left:auto">Clear all</button>
        </div>

        <table class="kf-table" id="fpd-table">
            <thead><tr><th>#</th><th>Frame</th><th>Landmark</th><th>dx</th><th>dy</th><th></th></tr></thead>
            <tbody></tbody>
        </table>

        <div class="row">
            <label>Ease</label>
            <select id="fpd-ease">${EASES.map(e =>
                `<option value="${e}"${e === state.ease ? " selected" : ""}>${e}</option>`).join("")}</select>
            <label style="margin-left:14px">Outside span</label>
            <select id="fpd-extrap">${EXTRAPS.map(x =>
                `<option value="${x}"${x === state.extrapolate ? " selected" : ""}>${x}</option>`).join("")}</select>
        </div>

        <div class="row" style="margin-top:10px">
            <label>JSON preview</label>
            <span style="color:var(--c2c-overlay2);font-size:11px">edit here if you prefer; auto-syncs on Save</span>
        </div>
        <textarea id="fpd-json" spellcheck="false"></textarea>
        <div class="err" id="fpd-err"></div>

        <div class="btn-row">
            <button class="btn" id="fpd-cancel">Cancel</button>
            <button class="btn primary" id="fpd-save">Save → node</button>
        </div>
    `;
    backdrop.appendChild(modal);
    document.body.appendChild(backdrop);

    // ── Shortcut buttons ──
    const shortcutsDiv = modal.querySelector("#fpd-shortcuts");
    for (const s of MP_SHORTCUTS) {
        const b = document.createElement("button");
        b.textContent = `${s.label} (${s.idx})`;
        b.dataset.idx = String(s.idx);
        b.addEventListener("click", () => {
            modal.querySelector("#fpd-idx").value = s.idx;
            _refreshIdxName();
            _highlightShortcut(s.idx);
        });
        shortcutsDiv.appendChild(b);
    }

    function _highlightShortcut(idx) {
        shortcutsDiv.querySelectorAll("button").forEach(b =>
            b.classList.toggle("active", Number(b.dataset.idx) === idx));
    }
    function _refreshIdxName() {
        const idx = Number(modal.querySelector("#fpd-idx").value);
        const hit = MP_SHORTCUTS.find(s => s.idx === idx);
        modal.querySelector("#fpd-idx-name").textContent = hit ? `(${hit.label})` : "";
        _highlightShortcut(idx);
    }
    _refreshIdxName();

    // ── Slider ↔ number two-way bind ──
    function _bind(numId, sliderId) {
        const n = modal.querySelector(numId);
        const s = modal.querySelector(sliderId);
        n.addEventListener("input", () => { s.value = n.value; });
        s.addEventListener("input", () => { n.value = s.value; });
    }
    _bind("#fpd-frame", "#fpd-frame-slider");
    _bind("#fpd-dx",    "#fpd-dx-slider");
    _bind("#fpd-dy",    "#fpd-dy-slider");
    modal.querySelector("#fpd-idx").addEventListener("input", _refreshIdxName);

    // ── Table renderer ──
    function _renderTable() {
        const tb = modal.querySelector("#fpd-table tbody");
        tb.innerHTML = "";
        const flat = [];
        for (const kf of workingKfs) {
            const fr = Number(kf.frame ?? 0);
            for (const [idxStr, dxy] of Object.entries(kf.deltas || {})) {
                if (!Array.isArray(dxy) || dxy.length < 2) continue;
                flat.push({ frame: fr, idx: Number(idxStr), dx: Number(dxy[0]), dy: Number(dxy[1]) });
            }
        }
        flat.sort((a, b) => a.frame - b.frame || a.idx - b.idx);
        flat.forEach((e, i) => {
            const name = MP_SHORTCUTS.find(s => s.idx === e.idx)?.label;
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>${i + 1}</td>
                <td>${e.frame}</td>
                <td>${e.idx}${name ? ` <span style="color:var(--c2c-overlay2)">(${name})</span>` : ""}</td>
                <td>${e.dx.toFixed(3)}</td>
                <td>${e.dy.toFixed(3)}</td>
                <td><button class="del" data-frame="${e.frame}" data-idx="${e.idx}">✕</button></td>
            `;
            tb.appendChild(tr);
        });
        tb.querySelectorAll(".del").forEach(btn => {
            btn.addEventListener("click", () => {
                const fr = Number(btn.dataset.frame);
                const ix = String(btn.dataset.idx);
                for (const kf of workingKfs) {
                    if (Number(kf.frame) === fr && kf.deltas && ix in kf.deltas) {
                        delete kf.deltas[ix];
                    }
                }
                // Drop empty keyframes
                workingKfs = workingKfs.filter(k => Object.keys(k.deltas || {}).length > 0);
                _renderTable();
                _refreshJson();
            });
        });
    }

    function _currentPayload() {
        return {
            keyframes: workingKfs,
            ease: modal.querySelector("#fpd-ease").value,
            extrapolate: modal.querySelector("#fpd-extrap").value,
        };
    }
    function _refreshJson() {
        modal.querySelector("#fpd-json").value =
            JSON.stringify(_currentPayload(), null, 2);
        modal.querySelector("#fpd-err").textContent = "";
    }

    // ── Add / replace ──
    modal.querySelector("#fpd-add").addEventListener("click", () => {
        const frame = Math.max(0, Math.round(Number(modal.querySelector("#fpd-frame").value)));
        const idx   = Math.max(0, Math.round(Number(modal.querySelector("#fpd-idx").value)));
        const dx    = Number(modal.querySelector("#fpd-dx").value);
        const dy    = Number(modal.querySelector("#fpd-dy").value);
        if (!isFinite(dx) || !isFinite(dy)) {
            modal.querySelector("#fpd-err").textContent = "dx/dy must be numbers.";
            return;
        }
        // Find/create the keyframe for this frame
        let kf = workingKfs.find(k => Number(k.frame) === frame);
        if (!kf) { kf = { frame, deltas: {} }; workingKfs.push(kf); }
        if (!kf.deltas) kf.deltas = {};
        kf.deltas[String(idx)] = [dx, dy];
        _renderTable();
        _refreshJson();
    });

    modal.querySelector("#fpd-clear").addEventListener("click", async () => {
        if (!workingKfs.length) return;
        if (!(await c2cConfirm("Remove all keyframes?"))) return;
        workingKfs = [];
        _renderTable();
        _refreshJson();
    });

    modal.querySelector("#fpd-ease").addEventListener("change", _refreshJson);
    modal.querySelector("#fpd-extrap").addEventListener("change", _refreshJson);

    // ── Allow direct JSON edits on save ──
    function _close() { backdrop.remove(); }
    modal.querySelector("#fpd-cancel").addEventListener("click", _close);
    backdrop.addEventListener("click", (e) => { if (e.target === backdrop) _close(); });

    modal.querySelector("#fpd-save").addEventListener("click", () => {
        // Prefer the text-area content if the user edited it — it's the
        // source of truth. Validate before writing back.
        const raw = modal.querySelector("#fpd-json").value;
        let parsed;
        try {
            parsed = JSON.parse(raw);
        } catch (e) {
            modal.querySelector("#fpd-err").textContent =
                "JSON parse error: " + e.message;
            return;
        }
        if (!parsed || typeof parsed !== "object" || !Array.isArray(parsed.keyframes)) {
            modal.querySelector("#fpd-err").textContent =
                "Top-level object must include a 'keyframes' array.";
            return;
        }
        widget.value = JSON.stringify(parsed, null, 2);
        if (typeof widget.callback === "function") {
            try { widget.callback(widget.value, app.canvas, node); } catch { /* ignore */ }
        }
        app.canvas?.setDirty?.(true, true);
        _close();
    });

    _renderTable();
    _refreshJson();
}

// ─── ComfyUI extension registration ─────────────────────────────────────
app.registerExtension({
    name: "C2C.FacePoseDeltaEditor",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "FacePoseDeltaCoreMEC") return;

        const orig = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = orig?.apply(this, arguments);
            const widget = (this.widgets || []).find(
                w => w.name === "keyframe_edits_json");
            if (!widget) return r;

            // Inject a button widget right under the JSON textarea
            this.addWidget(
                "button",
                "🖐 Edit landmark deltas…",
                null,
                () => _openEditor(this, widget),
            );
            return r;
        };
    },

    async setup() {
        console.log("[MEC.FacePoseDelta] Editor loaded — button appears on FacePoseDeltaCoreMEC nodes.");
    },
});
