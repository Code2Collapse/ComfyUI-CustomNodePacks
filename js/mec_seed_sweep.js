/**
 * mec_seed_sweep.js — Phase 7: Seed Sweep UI
 *
 * Right-click any INT widget named "seed" (or similar) → "🎲 Sweep seeds…".
 * Prompts for a count, then queues N prompts with seed values current,
 * current+1, … current+N-1. Updates the widget value between each queue so
 * each run is reproducible.
 *
 * Setting:
 *   mec.seed_sweep.enabled — bool (default true)
 */

import { app } from "../../scripts/app.js";

const SEED_NAME_RE = /^(seed|noise_seed|rand_seed|sampling_seed)$/i;

const STYLE_ID = "mec-seed-sweep-style";
const TOAST_ID = "mec-seed-sweep-toast";

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
#${TOAST_ID} {
    position: fixed;
    bottom: 16px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 99995;
    padding: 6px 14px;
    border-radius: 14px;
    background: #1e1e2e;
    border: 1px solid #f9e2af;
    color: #f9e2af;
    font-size: 12px;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    display: none;
}
#${TOAST_ID}.visible { display: block; }
    `.trim();
    document.head.appendChild(style);
}

function _ensureToast() {
    let toast = document.getElementById(TOAST_ID);
    if (!toast) {
        toast = document.createElement("div");
        toast.id = TOAST_ID;
        document.body.appendChild(toast);
    }
    return toast;
}

function _showProgress(current, total) {
    const t = _ensureToast();
    if (current > total) {
        t.classList.remove("visible");
        return;
    }
    t.textContent = `🎲 Seed Sweep — running ${current} / ${total}`;
    t.classList.add("visible");
}

function _isSeedWidget(w) {
    if (!w || !w.name) return false;
    if (w.type !== "number" && w.type !== "INT" && w.type !== "int") {
        // Some ComfyUI builds use type "combo" with options like "fixed"
        if (typeof w.value !== "number") return false;
    }
    return SEED_NAME_RE.test(w.name);
}

async function _runSweep(node, widget, count) {
    const baseSeed = Number(widget.value) || 0;
    const total = Math.max(1, Math.min(64, count));
    for (let i = 0; i < total; i++) {
        _showProgress(i + 1, total);
        const newSeed = (baseSeed + i) >>> 0;          // uint32
        widget.value = newSeed;
        if (typeof widget.callback === "function") {
            try { widget.callback(newSeed, app.canvas, node); } catch { /* ignore */ }
        }
        // Mark canvas dirty so the widget redraws with the new value
        app.canvas?.setDirty?.(true, true);

        // Queue this prompt. queuePrompt is async — await so we serialize.
        try {
            await app.queuePrompt(0, 1);
        } catch (e) {
            console.warn("[MEC.SeedSweep] queuePrompt failed:", e);
            break;
        }
        // Tiny pause between queues so the server can register each one
        await new Promise(r => setTimeout(r, 50));
    }
    // Restore base seed (so the user's widget isn't left at the last value)
    widget.value = baseSeed;
    app.canvas?.setDirty?.(true, true);
    setTimeout(() => _showProgress(total + 1, total), 1500);
}

function _patchNodeContextMenu() {
    const orig = LGraphCanvas.prototype.getNodeMenuOptions;
    if (!orig || orig._mecSeedSweepPatched) return;

    LGraphCanvas.prototype.getNodeMenuOptions = function (node) {
        const opts = orig.call(this, node);

        const enabled = (() => {
            try { return app.ui.settings.getSettingValue("mec.seed_sweep.enabled", true); }
            catch { return true; }
        })();
        if (!enabled) return opts;

        const seedWidgets = (node.widgets || []).filter(_isSeedWidget);
        if (seedWidgets.length === 0) return opts;

        opts.push(null);
        for (const w of seedWidgets) {
            opts.push({
                content: `🎲 Sweep ${w.name} (8 runs)`,
                callback: () => _runSweep(node, w, 8),
            });
            opts.push({
                content: `🎲 Sweep ${w.name} — custom…`,
                callback: () => {
                    const ans = prompt(
                        `Sweep ${w.name}: how many runs? (current value = ${w.value})`,
                        "8",
                    );
                    if (!ans) return;
                    const n = parseInt(ans, 10);
                    if (!isFinite(n) || n <= 0) return;
                    _runSweep(node, w, n);
                },
            });
        }
        return opts;
    };
    LGraphCanvas.prototype.getNodeMenuOptions._mecSeedSweepPatched = true;
}

app.registerExtension({
    name: "MEC.SeedSweep",
    settings: [
        {
            id: "mec.seed_sweep.enabled",
            name: "Seed Sweep: right-click 'Sweep seeds…' menu",
            tooltip: "Queue multiple prompts with sequential seed values from a node's right-click menu.",
            type: "boolean",
            default: true,
        },
    ],
    async setup() {
        _injectStyle();
        _ensureToast();
        _patchNodeContextMenu();
        console.log("[MEC.SeedSweep] Loaded — right-click a node with a 'seed' widget.");
    },
});
