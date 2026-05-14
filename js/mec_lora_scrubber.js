/**
 * mec_lora_scrubber.js — Phase 9: LoRA Weight Scrubber
 *
 * Adds a "🎚 scrub" button next to the strength_model widget on LoRA-loader
 * nodes. Click & hold to scrub between 0.0 and 2.0; on mouseup, if
 * mec.lora_scrubber.auto_queue is true, queues a new prompt with the new
 * weight applied. Logs each change to console for parameter history.
 *
 * Setting:
 *   mec.lora_scrubber.enabled    — bool (default true)
 *   mec.lora_scrubber.auto_queue — bool (default false)
 */

import { app } from "../../scripts/app.js";

const LORA_NODE_RE = /lora.*loader|loraloader|loralightning|lcm.*lora/i;

const STYLE_ID = "mec-lora-scrubber-style";

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.mec-lora-scrubber-overlay {
    position: fixed;
    z-index: 99996;
    background: #181825;
    border: 1px solid #cba6f7;
    border-radius: 6px;
    padding: 8px 10px;
    color: #cdd6f4;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 11px;
    width: 260px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.6);
}
.mec-lora-scrubber-overlay .ls-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
    font-weight: 700;
    color: #cba6f7;
}
.mec-lora-scrubber-overlay .ls-close {
    background: transparent;
    border: none;
    color: #6c7086;
    cursor: pointer;
    font-size: 14px;
}
.mec-lora-scrubber-overlay input[type=range] {
    width: 100%;
}
.mec-lora-scrubber-overlay .ls-value {
    display: flex;
    justify-content: space-between;
    font-family: monospace;
    font-size: 12px;
    color: #a6e3a1;
    margin-top: 2px;
}
.mec-lora-scrubber-overlay .ls-actions {
    display: flex;
    gap: 6px;
    margin-top: 6px;
}
.mec-lora-scrubber-overlay button.ls-btn {
    background: #313244;
    border: 1px solid #45475a;
    color: #cdd6f4;
    border-radius: 4px;
    padding: 3px 10px;
    cursor: pointer;
    font-size: 11px;
}
.mec-lora-scrubber-overlay button.ls-btn:hover { border-color: #cba6f7; }
    `.trim();
    document.head.appendChild(style);
}

function _openScrubber(node, widget, mouseX, mouseY) {
    _injectStyle();
    document.querySelectorAll(".mec-lora-scrubber-overlay").forEach(el => el.remove());

    const original = Number(widget.value) || 0;
    const root = document.createElement("div");
    root.className = "mec-lora-scrubber-overlay";
    root.innerHTML = `
        <div class="ls-header">
            <span>🎚 ${widget.name}</span>
            <button class="ls-close">×</button>
        </div>
        <input type="range" min="0" max="2" step="0.01" value="${original}">
        <div class="ls-value">
            <span>0.00</span>
            <span class="ls-current">${original.toFixed(2)}</span>
            <span>2.00</span>
        </div>
        <div class="ls-actions">
            <button class="ls-btn ls-apply">Apply</button>
            <button class="ls-btn ls-queue">Apply + Queue</button>
            <button class="ls-btn ls-reset">Reset</button>
        </div>
    `;
    document.body.appendChild(root);

    const W = 260;
    const H = 130;
    let x = mouseX + 10;
    let y = mouseY + 10;
    if (x + W > window.innerWidth)  x = mouseX - W - 10;
    if (y + H > window.innerHeight) y = window.innerHeight - H - 8;
    root.style.left = Math.max(8, x) + "px";
    root.style.top  = Math.max(8, y) + "px";

    const slider  = root.querySelector("input[type=range]");
    const current = root.querySelector(".ls-current");

    const apply = (commit) => {
        const v = parseFloat(slider.value);
        widget.value = v;
        if (typeof widget.callback === "function") {
            try { widget.callback(v, app.canvas, node); } catch { /* ignore */ }
        }
        current.textContent = v.toFixed(2);
        app.canvas?.setDirty?.(true, true);
        if (commit) console.log(`[MEC.LoRAScrubber] ${node.title || node.type} · ${widget.name} → ${v.toFixed(3)}`);
    };

    slider.addEventListener("input", () => apply(false));
    root.querySelector(".ls-close").addEventListener("click", () => {
        widget.value = original;
        app.canvas?.setDirty?.(true, true);
        root.remove();
    });
    root.querySelector(".ls-apply").addEventListener("click", () => {
        apply(true);
        root.remove();
    });
    root.querySelector(".ls-queue").addEventListener("click", async () => {
        apply(true);
        root.remove();
        try { await app.queuePrompt(0, 1); }
        catch (e) { console.warn("[MEC.LoRAScrubber] queue failed:", e); }
    });
    root.querySelector(".ls-reset").addEventListener("click", () => {
        slider.value = "1.0";
        apply(false);
    });
}

function _isLoraNode(node) {
    return LORA_NODE_RE.test(node.type || node.comfyClass || "");
}

function _isStrengthWidget(w) {
    if (!w || !w.name) return false;
    return /^strength_(model|clip)$/i.test(w.name) ||
           /^lora_strength/i.test(w.name) ||
           w.name.toLowerCase() === "strength";
}

function _patchNodeMenu() {
    const orig = LGraphCanvas.prototype.getNodeMenuOptions;
    if (!orig || orig._mecLoraScrubPatched) return;
    LGraphCanvas.prototype.getNodeMenuOptions = function (node) {
        const opts = orig.call(this, node);

        const enabled = (() => {
            try { return app.ui.settings.getSettingValue("mec.lora_scrubber.enabled", true); }
            catch { return true; }
        })();
        if (!enabled) return opts;
        if (!_isLoraNode(node)) return opts;

        const strengths = (node.widgets || []).filter(_isStrengthWidget);
        if (strengths.length === 0) return opts;

        opts.push(null);
        for (const w of strengths) {
            opts.push({
                content: `🎚 Scrub ${w.name}`,
                callback: () => {
                    const rect = this.canvas.getBoundingClientRect();
                    _openScrubber(
                        node, w,
                        rect.left + rect.width / 2,
                        rect.top + rect.height / 2,
                    );
                },
            });
        }
        return opts;
    };
    LGraphCanvas.prototype.getNodeMenuOptions._mecLoraScrubPatched = true;
}

app.registerExtension({
    name: "MEC.LoRAScrubber",
    settings: [
        {
            id: "mec.lora_scrubber.enabled",
            name: "LoRA Scrubber: enabled",
            tooltip: "Right-click a LoRA loader → 'Scrub strength_model' for an inline slider.",
            type: "boolean",
            defaultValue: true,
        },
    ],
    async setup() {
        _injectStyle();
        _patchNodeMenu();
        console.log("[MEC.LoRAScrubber] Loaded.");
    },
});
