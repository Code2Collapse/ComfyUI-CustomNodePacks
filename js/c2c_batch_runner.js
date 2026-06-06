/**
 * mec_batch_runner.js — Flagship §17.4: Batch Runner.
 *
 * The single highest-leverage beginner feature in the pack (per
 * `third_party/ideas_summary.md` gap matrix). Lets the user queue a
 * cartesian product OR zipped sweep over several widget values without
 * editing JSON or scripting:
 *
 *   • Right-click any node → "📦 Batch Run…".
 *   • A modal opens listing widgets that hold simple values (numbers,
 *     strings, combos). For each widget pick a list of values:
 *         - "1,2,3"           — explicit
 *         - "1..10"           — inclusive integer range
 *         - "0:1:0.25"        — start:stop:step (incl.) for floats
 *         - "red|blue|green"  — pipe-separated strings
 *   • Pick combine mode:
 *         - "cross"  — full cartesian product
 *         - "zip"    — pairwise (lists must be equal length)
 *   • Click "Run" — the runner sequentially:
 *         1. sets each widget value,
 *         2. fires widget.callback so dependent UI updates,
 *         3. awaits `app.queuePrompt(0, 1)`,
 *         4. shows a progress toast and an in-modal log.
 *   • "Stop" interrupts mid-sweep without breaking the current prompt.
 *
 * Restoration of pre-run widget values happens on completion *or* stop.
 *
 * No dependencies on any other MEC node — works against any ComfyUI node
 * the user picks.
 *
 * Settings:
 *   mec.batch_runner.enabled — bool (default true)
 *   mec.batch_runner.max_runs — int (default 256, hard ceiling 4096)
 *
 * License: Apache-2.0
 */

import { app } from "../../scripts/app.js";
import { c2cConfirm } from "./_c2c_dialog.js";

const STYLE_ID  = "mec-batch-style";
const MODAL_ID  = "mec-batch-modal";
const TOAST_ID  = "mec-batch-toast";

const SETTING_ENABLED  = "mec.batch_runner.enabled";
const SETTING_MAX_RUNS = "mec.batch_runner.max_runs";

// ───────────────────────────────────────────────────────────────────────
// Value-list parser
// ───────────────────────────────────────────────────────────────────────
/**
 * Parse a textual spec into a concrete array of values.
 *
 *   "1,2,3"             → [1, 2, 3]                       (numbers)
 *   "a,b,c"             → ["a","b","c"]                   (strings)
 *   "1..5"              → [1, 2, 3, 4, 5]                 (inclusive int)
 *   "0:1:0.25"          → [0, 0.25, 0.5, 0.75, 1.0]       (incl. float)
 *   "red|blue|green"    → ["red","blue","green"]
 *
 * `coerce` selects the target type when the spec is ambiguous (number vs
 * string). Returns { values, error } — error is non-empty on parse fail.
 */
export function parseValueSpec(spec, coerce /* "number" | "string" */) {
    const out = { values: [], error: "" };
    const s = String(spec ?? "").trim();
    if (!s) { out.error = "empty spec"; return out; }

    // Pipe-separated strings always win
    if (s.includes("|")) {
        out.values = s.split("|").map(x => x.trim()).filter(x => x.length);
        if (coerce === "number") out.values = out.values.map(Number);
        return out;
    }

    // start:stop:step (incl.) — only meaningful for numbers
    const colonMatch = /^(-?\d+(?:\.\d+)?)\s*:\s*(-?\d+(?:\.\d+)?)\s*:\s*(-?\d+(?:\.\d+)?)$/.exec(s);
    if (colonMatch) {
        const a = parseFloat(colonMatch[1]);
        const b = parseFloat(colonMatch[2]);
        const step = parseFloat(colonMatch[3]);
        if (step === 0) { out.error = "step cannot be 0"; return out; }
        const dir = Math.sign(b - a);
        if (dir !== 0 && Math.sign(step) !== dir) { out.error = "step sign disagrees with range"; return out; }
        if (a === b) { out.values = [a]; return out; }
        const vs = [];
        for (let v = a; (step > 0 ? v <= b + 1e-9 : v >= b - 1e-9); v += step) {
            vs.push(Number(v.toFixed(10)));
            if (vs.length > 100000) { out.error = "range too large"; return out; }
        }
        out.values = vs;
        return out;
    }

    // ".." inclusive integer range
    const rangeMatch = /^(-?\d+)\s*\.\.\s*(-?\d+)$/.exec(s);
    if (rangeMatch) {
        const a = parseInt(rangeMatch[1], 10);
        const b = parseInt(rangeMatch[2], 10);
        const dir = a <= b ? 1 : -1;
        for (let v = a; dir > 0 ? v <= b : v >= b; v += dir) out.values.push(v);
        if (out.values.length > 100000) { out.error = "range too large"; out.values = []; return out; }
        return out;
    }

    // Comma-separated
    const parts = s.split(",").map(x => x.trim()).filter(x => x.length);
    if (!parts.length) { out.error = "no values parsed"; return out; }
    if (coerce === "number") {
        const nums = parts.map(Number);
        if (nums.some(n => !isFinite(n))) { out.error = "non-numeric value"; return out; }
        out.values = nums;
    } else if (coerce === "string") {
        out.values = parts;
    } else {
        // Auto-detect: if every part parses as finite number, numbers; else strings
        const nums = parts.map(Number);
        out.values = nums.every(n => isFinite(n) && parts[String(nums.indexOf(n))] !== "")
            ? nums : parts;
        // Stricter: if any part is non-numeric, use strings
        if (parts.some(p => !isFinite(Number(p)))) out.values = parts;
        else out.values = parts.map(Number);
    }
    return out;
}

// ───────────────────────────────────────────────────────────────────────
// Combinator
// ───────────────────────────────────────────────────────────────────────
/**
 * Combine per-axis value arrays into a list of assignments (each
 * assignment is a list of {axisIdx, value} entries).
 *
 *   cartesianProduct([[1,2],[a,b]]) → 4 assignments
 *   zip([[1,2,3],[a,b,c]])          → 3 assignments (error if lengths differ)
 */
export function combineAxes(axes, mode) {
    if (!axes.length) return [];
    if (mode === "zip") {
        const n = axes[0].length;
        if (axes.some(a => a.length !== n)) {
            return { error: `zip requires all axes to have the same length; got [${axes.map(a => a.length).join(", ")}]` };
        }
        const out = [];
        for (let i = 0; i < n; i++) {
            out.push(axes.map((a, ax) => ({ axisIdx: ax, value: a[i] })));
        }
        return out;
    }
    // Cartesian product
    let acc = [[]];
    for (let ax = 0; ax < axes.length; ax++) {
        const next = [];
        for (const prefix of acc) {
            for (const v of axes[ax]) {
                next.push(prefix.concat([{ axisIdx: ax, value: v }]));
            }
        }
        acc = next;
    }
    return acc;
}

// ───────────────────────────────────────────────────────────────────────
// CSS + toast helpers
// ───────────────────────────────────────────────────────────────────────
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
    z-index: var(--c2c-z-modal); width: 720px; max-width: 96vw; max-height: 92vh;
    background: var(--c2c-bg); color: var(--c2c-fg); border: 1px solid var(--c2c-border);
    border-radius: 10px; box-shadow: 0 8px 32px rgba(0,0,0,0.6);
    font: 13px/1.45 system-ui, sans-serif; padding: 18px 20px;
    overflow-y: auto;
}
#${MODAL_ID} h3 { margin: 0 0 4px; color: var(--c2c-lavender); font-size: 16px; }
#${MODAL_ID} p.hint { margin: 0 0 14px; color: var(--c2c-overlay2); font-size: 11px; }
#${MODAL_ID} .axis-list { display: flex; flex-direction: column; gap: 6px; margin: 8px 0 12px; }
#${MODAL_ID} .axis {
    display: flex; gap: 6px; align-items: center;
    background: var(--c2c-bg2); border: 1px solid var(--c2c-border); border-radius: 5px; padding: 6px 8px;
}
#${MODAL_ID} .axis label { color: var(--c2c-sub); font-size: 12px; min-width: 100px; }
#${MODAL_ID} .axis input[type="text"] {
    flex: 1; background: var(--c2c-bg3); color: var(--c2c-fg); border: 1px solid var(--c2c-surface1);
    border-radius: 4px; padding: 4px 6px; font: 11px ui-monospace, monospace;
}
#${MODAL_ID} .axis .count { color: var(--c2c-green); font-size: 11px; min-width: 60px; text-align: right; }
#${MODAL_ID} .axis .count.bad { color: var(--c2c-red); }
#${MODAL_ID} .axis .rm {
    background: transparent; color: var(--c2c-red); border: 1px solid var(--c2c-red);
    border-radius: 3px; padding: 1px 6px; cursor: pointer; font: 11px inherit;
}
#${MODAL_ID} .axis .rm:hover { background: var(--c2c-red); color: var(--c2c-bg3); }
#${MODAL_ID} .row { display: flex; gap: 8px; align-items: center; margin: 8px 0; }
#${MODAL_ID} .row label { min-width: 110px; color: var(--c2c-sub); font-size: 12px; }
#${MODAL_ID} select, #${MODAL_ID} input[type="number"] {
    background: var(--c2c-bg2); color: var(--c2c-fg); border: 1px solid var(--c2c-surface1);
    border-radius: 4px; padding: 4px 6px; font: inherit;
}
#${MODAL_ID} .total {
    margin: 6px 0; padding: 6px 10px; background: var(--c2c-bg2);
    border-left: 3px solid var(--c2c-blue); border-radius: 4px; font-size: 12px;
}
#${MODAL_ID} .total.warn { border-left-color: var(--c2c-yellow); color: var(--c2c-yellow); }
#${MODAL_ID} .total.bad  { border-left-color: var(--c2c-red); color: var(--c2c-red); }
#${MODAL_ID} .log {
    background: var(--c2c-bg3); border: 1px solid var(--c2c-border); border-radius: 4px;
    padding: 6px 8px; height: 110px; overflow-y: auto;
    font: 11px ui-monospace, monospace; color: var(--c2c-sub);
}
#${MODAL_ID} .log .ok  { color: var(--c2c-green); }
#${MODAL_ID} .log .err { color: var(--c2c-red); }
#${MODAL_ID} .btn-row { display: flex; gap: 8px; margin-top: 14px; }
#${MODAL_ID} .btn-row .spacer { flex: 1; }
#${MODAL_ID} .btn {
    background: var(--c2c-surface1); color: var(--c2c-fg); border: 1px solid var(--c2c-surface2);
    border-radius: 5px; padding: 6px 14px; cursor: pointer; font: inherit;
}
#${MODAL_ID} .btn:hover:not(:disabled) { background: var(--c2c-surface2); }
#${MODAL_ID} .btn:disabled { opacity: 0.5; cursor: not-allowed; }
#${MODAL_ID} .btn.primary { background: var(--c2c-blue); color: var(--c2c-bg3); border-color: var(--c2c-blue); }
#${MODAL_ID} .btn.primary:hover:not(:disabled) { background: var(--c2c-lavender); }
#${MODAL_ID} .btn.stop { background: var(--c2c-red); color: var(--c2c-bg3); border-color: var(--c2c-red); }
#${MODAL_ID} .btn.stop:hover:not(:disabled) { background: var(--c2c-maroon); }
#${TOAST_ID} {
    position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%);
    z-index: var(--c2c-z-modal); padding: 6px 14px; border-radius: 14px;
    background: var(--c2c-bg); border: 1px solid var(--c2c-blue); color: var(--c2c-blue);
    font-size: 12px; font-weight: 600;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5); display: none;
}
#${TOAST_ID}.visible { display: block; }
    `.trim();
    document.head.appendChild(s);
}
function _ensureToast() {
    let t = document.getElementById(TOAST_ID);
    if (!t) { t = document.createElement("div"); t.id = TOAST_ID; document.body.appendChild(t); }
    return t;
}
function _showToast(text) {
    const t = _ensureToast();
    if (!text) { t.classList.remove("visible"); return; }
    t.textContent = text; t.classList.add("visible");
}

// ───────────────────────────────────────────────────────────────────────
// Widget enumeration — which widgets are "sweepable"?
// ───────────────────────────────────────────────────────────────────────
function _sweepableWidgets(node) {
    return (node.widgets || []).filter(w => {
        if (!w || !w.name) return false;
        // Buttons are not sweepable
        if (w.type === "button") return false;
        // Output/preview widgets aren't either
        if (typeof w.value === "undefined") return false;
        // Skip multi-line text blobs (would be too large to sweep over)
        if (w.options?.multiline) return false;
        return true;
    });
}

function _widgetCoerceHint(w) {
    if (typeof w.value === "number") return "number";
    if (w.type === "combo" || Array.isArray(w.options?.values)) return "string";
    return "auto";
}

// ───────────────────────────────────────────────────────────────────────
// Runner
// ───────────────────────────────────────────────────────────────────────
let _activeRun = null;  // { stopRequested: bool }

function _openModal(node) {
    _injectStyle();

    let backdrop = document.getElementById(`${MODAL_ID}-backdrop`);
    if (backdrop) backdrop.remove();
    backdrop = document.createElement("div");
    backdrop.id = `${MODAL_ID}-backdrop`;
    backdrop.classList.add("visible");

    const modal = document.createElement("div");
    modal.id = MODAL_ID;

    const sweepable = _sweepableWidgets(node);
    if (!sweepable.length) {
        modal.innerHTML = `
            <h3>📦 Batch Run</h3>
            <p class="hint">This node has no widgets that can be swept (no numbers, strings, or combos available).</p>
            <div class="btn-row"><div class="spacer"></div><button class="btn primary" id="mb-close">Close</button></div>
        `;
        backdrop.appendChild(modal);
        document.body.appendChild(backdrop);
        modal.querySelector("#mb-close").addEventListener("click", () => backdrop.remove());
        return;
    }

    const widgetOptions = sweepable
        .map(w => `<option value="${w.name}">${w.name}  (${typeof w.value === "number" ? "number" : "text"}: ${String(w.value).slice(0, 24)})</option>`)
        .join("");

    modal.innerHTML = `
        <h3>📦 Batch Run — <span style="color:var(--c2c-fg)">${node.title || node.type}</span></h3>
        <p class="hint">Set lists of values for one or more widgets. Each row supports
            <code>1,2,3</code>, <code>1..10</code>, <code>0:1:0.25</code>, or <code>red|blue|green</code>.</p>

        <div class="axis-list" id="mb-axes"></div>

        <div class="row">
            <button class="btn" id="mb-add">＋ Add axis</button>
            <label style="margin-left:18px">Combine</label>
            <select id="mb-combine">
                <option value="cross">cross (cartesian product)</option>
                <option value="zip">zip (pairwise)</option>
            </select>
            <label style="margin-left:18px">Delay (ms)</label>
            <input type="number" id="mb-delay" value="50" min="0" max="5000" step="10">
        </div>

        <div class="total" id="mb-total">Total runs: 0</div>
        <div class="log" id="mb-log"></div>

        <div class="btn-row">
            <button class="btn" id="mb-close">Close</button>
            <div class="spacer"></div>
            <button class="btn stop" id="mb-stop" disabled>Stop</button>
            <button class="btn primary" id="mb-run">▶ Run batch</button>
        </div>
    `;
    backdrop.appendChild(modal);
    document.body.appendChild(backdrop);

    const axesDiv = modal.querySelector("#mb-axes");
    const totalEl = modal.querySelector("#mb-total");
    const logEl   = modal.querySelector("#mb-log");
    const runBtn  = modal.querySelector("#mb-run");
    const stopBtn = modal.querySelector("#mb-stop");

    const maxRuns = (() => {
        try {
            const v = app.ui.settings.getSettingValue(SETTING_MAX_RUNS, 256);
            return Math.max(1, Math.min(4096, Number(v) || 256));
        } catch { return 256; }
    })();

    function _addAxis(preselectName) {
        const row = document.createElement("div");
        row.className = "axis";
        row.innerHTML = `
            <label>Widget</label>
            <select class="w">${widgetOptions}</select>
            <input type="text" class="spec" placeholder="e.g. 1,2,3  or  1..5  or  0:1:0.25">
            <span class="count">0</span>
            <button class="rm" title="Remove axis">✕</button>
        `;
        axesDiv.appendChild(row);
        if (preselectName) row.querySelector(".w").value = preselectName;
        row.querySelector(".rm").addEventListener("click", () => { row.remove(); _refreshTotal(); });
        row.querySelector(".w").addEventListener("change", _refreshTotal);
        row.querySelector(".spec").addEventListener("input", _refreshTotal);
        _refreshTotal();
    }

    function _allAxes() {
        const rows = Array.from(axesDiv.querySelectorAll(".axis"));
        const out = [];
        for (const row of rows) {
            const name  = row.querySelector(".w").value;
            const spec  = row.querySelector(".spec").value;
            const widget = sweepable.find(w => w.name === name);
            if (!widget) continue;
            const coerce = _widgetCoerceHint(widget);
            const parsed = parseValueSpec(spec, coerce === "auto" ? undefined : coerce);
            const countEl = row.querySelector(".count");
            if (parsed.error) {
                countEl.textContent = "err"; countEl.classList.add("bad");
                out.push({ widget, values: [], error: parsed.error });
            } else {
                countEl.textContent = String(parsed.values.length);
                countEl.classList.toggle("bad", parsed.values.length === 0);
                out.push({ widget, values: parsed.values, error: "" });
            }
        }
        return out;
    }

    function _refreshTotal() {
        const axes = _allAxes();
        if (!axes.length || axes.some(a => a.error || !a.values.length)) {
            totalEl.textContent = axes.length === 0
                ? "Total runs: 0  (add an axis to begin)"
                : "Total runs: ? — fix axis errors";
            totalEl.classList.toggle("warn", true);
            totalEl.classList.toggle("bad", axes.some(a => a.error));
            runBtn.disabled = true;
            return;
        }
        const mode = modal.querySelector("#mb-combine").value;
        let total;
        if (mode === "zip") {
            const n = axes[0].values.length;
            if (axes.some(a => a.values.length !== n)) {
                totalEl.textContent = `Total runs: invalid — zip needs equal-length axes [${axes.map(a => a.values.length).join(", ")}]`;
                totalEl.classList.add("bad"); runBtn.disabled = true; return;
            }
            total = n;
        } else {
            total = axes.reduce((acc, a) => acc * a.values.length, 1);
        }
        const overLimit = total > maxRuns;
        totalEl.textContent = `Total runs: ${total}${overLimit ? `  ⚠ exceeds limit (${maxRuns}) — raise mec.batch_runner.max_runs to proceed` : ""}`;
        totalEl.classList.toggle("warn", overLimit);
        totalEl.classList.toggle("bad", false);
        runBtn.disabled = overLimit || total === 0;
    }
    modal.querySelector("#mb-combine").addEventListener("change", _refreshTotal);
    modal.querySelector("#mb-add").addEventListener("click", () => _addAxis());

    // Start with one axis selecting the first sweepable widget
    _addAxis(sweepable[0].name);

    function _log(msg, cls = "") {
        const line = document.createElement("div");
        if (cls) line.className = cls;
        line.textContent = msg;
        logEl.appendChild(line);
        logEl.scrollTop = logEl.scrollHeight;
    }

    async function _runBatch() {
        const axes = _allAxes();
        if (!axes.length || axes.some(a => a.error || !a.values.length)) {
            _log("Aborted — axis errors.", "err"); return;
        }
        const mode = modal.querySelector("#mb-combine").value;
        const assignments = combineAxes(axes.map(a => a.values), mode);
        if (assignments.error) { _log(`Aborted — ${assignments.error}`, "err"); return; }
        if (!assignments.length) { _log("Aborted — empty plan.", "err"); return; }
        if (assignments.length > maxRuns) {
            _log(`Aborted — ${assignments.length} runs exceeds limit (${maxRuns}).`, "err"); return;
        }

        // Snapshot original widget values for restoration on stop / completion
        const original = axes.map(a => ({ widget: a.widget, value: a.widget.value }));

        const delayMs = Math.max(0, Math.min(5000, Number(modal.querySelector("#mb-delay").value) || 0));

        _activeRun = { stopRequested: false };
        runBtn.disabled = true;
        stopBtn.disabled = false;
        modal.querySelector("#mb-close").disabled = true;

        const total = assignments.length;
        _log(`▶ Running ${total} prompts (${mode})…`, "ok");

        for (let i = 0; i < total; i++) {
            if (_activeRun.stopRequested) { _log(`■ Stopped after ${i}/${total}.`, "err"); break; }
            const plan = assignments[i];
            // Apply axis values to widgets
            const summary = [];
            for (const step of plan) {
                const ax = axes[step.axisIdx];
                ax.widget.value = step.value;
                if (typeof ax.widget.callback === "function") {
                    try { ax.widget.callback(step.value, app.canvas, node); } catch { /* ignore */ }
                }
                summary.push(`${ax.widget.name}=${step.value}`);
            }
            app.canvas?.setDirty?.(true, true);
            _showToast(`📦 Batch ${i + 1} / ${total} — ${summary.join("  ")}`);
            _log(`[${i + 1}/${total}] ${summary.join("  ")}`);

            try {
                await app.queuePrompt(0, 1);
            } catch (e) {
                _log(`  ✗ queuePrompt failed: ${e?.message || e}`, "err");
                // continue — let the user decide whether to stop
            }

            if (delayMs > 0) await new Promise(r => setTimeout(r, delayMs));
        }

        // Restore original widget values
        for (const o of original) {
            o.widget.value = o.value;
            if (typeof o.widget.callback === "function") {
                try { o.widget.callback(o.value, app.canvas, node); } catch { /* ignore */ }
            }
        }
        app.canvas?.setDirty?.(true, true);
        _showToast("");
        _log(_activeRun.stopRequested ? "✓ Original values restored." : "✓ Batch finished. Original values restored.", "ok");

        _activeRun = null;
        runBtn.disabled = false;
        stopBtn.disabled = true;
        modal.querySelector("#mb-close").disabled = false;
    }

    runBtn.addEventListener("click", _runBatch);
    stopBtn.addEventListener("click", () => {
        if (_activeRun) _activeRun.stopRequested = true;
        stopBtn.disabled = true;
        _log("Stop requested — finishing current prompt first…", "err");
    });

    function _close() {
        if (_activeRun) {
            c2cConfirm("A batch is running. Stop it and close?").then((ok) => {
                if (!ok) return;
                _activeRun.stopRequested = true;
                backdrop.remove();
                _showToast("");
            });
            return;
        }
        backdrop.remove();
        _showToast("");
    }
    modal.querySelector("#mb-close").addEventListener("click", _close);
    backdrop.addEventListener("click", (e) => { if (e.target === backdrop) _close(); });
}

// ───────────────────────────────────────────────────────────────────────
// Context-menu integration
// ───────────────────────────────────────────────────────────────────────
function _patchNodeContextMenu() {
    const orig = LGraphCanvas.prototype.getNodeMenuOptions;
    if (!orig || orig._mecBatchPatched) return;

    LGraphCanvas.prototype.getNodeMenuOptions = function (node) {
        const opts = orig.call(this, node);
        let enabled = true;
        try { enabled = app.ui.settings.getSettingValue(SETTING_ENABLED, true); } catch { /* default */ }
        if (!enabled) return opts;
        if (!_sweepableWidgets(node).length) return opts;

        opts.push(null);
        opts.push({
            content: "📦 Batch Run…",
            callback: () => _openModal(node),
        });
        return opts;
    };
    LGraphCanvas.prototype.getNodeMenuOptions._mecBatchPatched = true;
}

// ───────────────────────────────────────────────────────────────────────
// Extension registration
// ───────────────────────────────────────────────────────────────────────
app.registerExtension({
    name: "C2C.BatchRunner",
    settings: [
        {
            id: SETTING_ENABLED,
            name: "Batch Runner: right-click '📦 Batch Run…' menu",
            tooltip: "Sweep one or more widgets over a list of values and auto-queue all combinations.",
            type: "boolean",
            default: true,
        },
        {
            id: SETTING_MAX_RUNS,
            name: "Batch Runner: max runs per batch",
            tooltip: "Hard ceiling on how many prompts a single batch can queue (1–4096).",
            type: "number",
            default: 256,
            attrs: { min: 1, max: 4096, step: 1 },
        },
    ],
    async setup() {
        _injectStyle();
        _ensureToast();
        _patchNodeContextMenu();
        console.log("[MEC.BatchRunner] Loaded — right-click a node → '📦 Batch Run…'.");
    },
});
