/**
 * c2c_surprise_me.js — Surprise Me (god-level rebuild, 2026-05-27)
 *
 * Floating 🎰 button + right-click popover with 3 randomness profiles:
 *   • mild  — randomize seeds only (safe re-roll)
 *   • wild  — seeds + ±20% cfg + sampler swap from a compatible pool
 *   • chaos — seeds + cfg ±50% + steps ±50% + sampler/scheduler swap +
 *             AI-generated style tag appended to the positive prompt
 *
 * Features:
 *   1) Three randomness profiles (mild / wild / chaos) selectable per click
 *   2) Seed randomization (any widget whose name matches /seed/i)
 *   3) CFG nudge with bounds [1, 30]
 *   4) Steps nudge with bounds [4, 80]
 *   5) Sampler & scheduler swap from a known-good pool (preserves combo enum)
 *   6) AI style-tag injection (chaos only, guarded)
 *   7) Right-click button → profile popover; single-click uses default profile
 *   8) "Dry run" mode that mutates widgets but does NOT queue prompt
 *   9) Settings-driven enable/default-profile/spice; safe value-write via callback
 */

import { app } from "../../scripts/app.js";
import { streamAI } from "./_c2c_ai_client.js";

const BTN_ID    = "mec-surprise-btn";
const POP_ID    = "mec-surprise-pop";
const STYLE_ID  = "mec-surprise-style";

const SAMPLER_POOL    = ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "uni_pc", "heun", "lms"];
const SCHEDULER_POOL  = ["normal", "karras", "exponential", "sgm_uniform", "simple", "beta"];

function _setting(key, def) { try { return app.ui.settings.getSettingValue(key, def); } catch { return def; } }
function _enabled()         { return !!_setting("mec.surprise_me.enabled", true); }
function _defaultProfile()  { return _setting("mec.surprise_me.profile", "mild"); }
function _spiceAllowed()    { return !!_setting("mec.surprise_me.spice_prompts", false); }
function _dryRun()          { return !!_setting("mec.surprise_me.dry_run", false); }

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
#${BTN_ID} {
    position: fixed; bottom: 294px; right: 16px;
    z-index: var(--c2c-z-hud, 1000);
    width: 38px; height: 38px; border-radius: 50%;
    background: var(--c2c-bg); border: 1px solid var(--c2c-surface1); color: var(--c2c-pink);
    font-size: 16px; cursor: pointer; box-shadow: 0 2px 8px rgba(0,0,0,0.5);
    display: flex; align-items: center; justify-content: center; transition: transform 0.15s;
}
#${BTN_ID}:hover { border-color: var(--c2c-pink); transform: rotate(15deg); }
#${BTN_ID}.spinning { animation: mec-surprise-spin 0.8s ease-out; }
@keyframes mec-surprise-spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(720deg); } }
#${BTN_ID} .badge {
    position: absolute; bottom: -2px; right: -2px; min-width: 14px; height: 14px;
    border-radius: 7px; background: var(--c2c-pink); color: var(--c2c-bg); font-size: 9px;
    font-weight: 700; padding: 0 3px; display: flex; align-items: center; justify-content: center;
    border: 1px solid var(--c2c-bg);
}

#${POP_ID} {
    position: fixed; z-index: calc(var(--c2c-z-hud, 1000) + 1);
    background: var(--c2c-bg); border: 1px solid var(--c2c-surface1); border-radius: 8px;
    padding: 8px; color: var(--c2c-fg); font-family: -apple-system, "Segoe UI", sans-serif;
    font-size: 12px; box-shadow: 0 6px 24px rgba(0,0,0,0.7); display: none; min-width: 200px;
}
#${POP_ID}.visible { display: block; }
#${POP_ID} .sm-title { color: var(--c2c-pink); font-weight: 700; margin-bottom: 6px; font-size: 12px; }
#${POP_ID} .sm-row { display: flex; align-items: center; gap: 6px; padding: 4px 6px; border-radius: 4px; cursor: pointer; }
#${POP_ID} .sm-row:hover { background: var(--c2c-surface0); }
#${POP_ID} .sm-row .sm-emoji { width: 22px; text-align: center; }
#${POP_ID} .sm-row .sm-name { flex: 1; font-weight: 600; }
#${POP_ID} .sm-row .sm-desc { font-size: 10px; color: var(--c2c-overlay0); }
#${POP_ID} .sm-row.active { border: 1px solid var(--c2c-pink); }
#${POP_ID} hr { border: none; border-top: 1px solid var(--c2c-surface0); margin: 6px 0; }
#${POP_ID} .sm-opt { display: flex; align-items: center; gap: 6px; padding: 4px 6px; font-size: 11px; color: var(--c2c-fg); }
#${POP_ID} .sm-opt input { accent-color: var(--c2c-pink); }
    `.trim();
    document.head.appendChild(style);
}

// ─────────────────────────────────────────────────────────────────────────────
// Mutations
// ─────────────────────────────────────────────────────────────────────────────

function _writeWidget(node, w, value) {
    w.value = value;
    if (typeof w.callback === "function") {
        try { w.callback(value, app.canvas, node); } catch {}
    }
}

function _randomizeSeeds() {
    const g = app.graph;
    if (!g || !g._nodes) return 0;
    let n = 0;
    for (const node of g._nodes) {
        for (const w of node.widgets || []) {
            if (!w || !w.name || !/seed/i.test(w.name) || typeof w.value !== "number") continue;
            _writeWidget(node, w, Math.floor(Math.random() * 0xFFFFFFFF));
            n++;
        }
    }
    return n;
}

function _nudgeNumeric(name, factor, lo, hi) {
    const g = app.graph;
    if (!g || !g._nodes) return 0;
    let n = 0;
    for (const node of g._nodes) {
        for (const w of node.widgets || []) {
            if (!w || !w.name || w.name.toLowerCase() !== name || typeof w.value !== "number") continue;
            const f = 1 + (Math.random() * 2 - 1) * factor; // ±factor
            let v = w.value * f;
            if (Number.isInteger(w.value)) v = Math.round(v);
            v = Math.min(hi, Math.max(lo, v));
            _writeWidget(node, w, v);
            n++;
        }
    }
    return n;
}

function _swapCombo(name, pool) {
    const g = app.graph;
    if (!g || !g._nodes) return 0;
    let n = 0;
    for (const node of g._nodes) {
        for (const w of node.widgets || []) {
            if (!w || !w.name || w.name.toLowerCase() !== name) continue;
            const opts = w.options?.values || w.values || null;
            const valid = Array.isArray(opts) ? pool.filter((p) => opts.includes(p)) : pool;
            if (!valid.length) continue;
            const pick = valid[Math.floor(Math.random() * valid.length)];
            if (pick === w.value) continue;
            _writeWidget(node, w, pick);
            n++;
        }
    }
    return n;
}

async function _generateStyleTag(original) {
    let buf = "";
    try {
        const r = await streamAI({
            feature: "surprise_style",
            sensitivity: "public",
            max_tokens: 30,
            temperature: 0.9,
            messages: [
                { role: "system", content: "You generate ONE short cinematic / lighting / style descriptor to append to an image generation prompt. Reply with ONLY the descriptor — 2 to 5 words, no quotes, no punctuation, no prefix." },
                { role: "user", content: `Current prompt: ${original.slice(0, 600)}\n\nReturn ONE descriptor only.` },
            ],
            onChunk: (c) => { buf += c; },
            onError: () => {},
            onDone: () => {},
        });
        let tag = (r?.text || buf || "").trim();
        tag = tag.replace(/^[`"'\s,.\-]+|[`"'\s,.\-]+$/g, "").split(/[\r\n]+/)[0].trim();
        if (!tag || tag.length > 60) return null;
        return tag;
    } catch { return null; }
}

async function _maybeSpicePrompt() {
    if (!_spiceAllowed()) return null;
    const g = app.graph;
    if (!g || !g._nodes) return null;
    for (const node of g._nodes) {
        if (!/cliptextencode/i.test(node.type || "")) continue;
        const w = (node.widgets || []).find((x) => x && (x.type === "customtext" || x.type === "string" || x.type === "STRING") && x.name === "text");
        if (!w) continue;
        const original = String(w.value || "").trim();
        if (!original) continue;
        if (/\b(worst|bad|low quality|ugly|blurry|nsfw)\b/i.test(original)) continue;
        const tag = await _generateStyleTag(original);
        if (!tag) return null;
        if (original.toLowerCase().includes(tag.toLowerCase())) return null;
        _writeWidget(node, w, `${original}, ${tag}`);
        return { tag, node: node.id };
    }
    return null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Profiles
// ─────────────────────────────────────────────────────────────────────────────

const PROFILES = {
    mild:  { emoji: "🎲", label: "Mild",  desc: "Seeds only" },
    wild:  { emoji: "🌪", label: "Wild",  desc: "Seeds + cfg ±20% + sampler swap" },
    chaos: { emoji: "💥", label: "Chaos", desc: "Seeds + cfg/steps ±50% + sampler + AI tag" },
};

async function _applyProfile(profile) {
    const stats = { profile, seeds: 0, cfg: 0, steps: 0, sampler: 0, scheduler: 0, spice: null };
    stats.seeds = _randomizeSeeds();
    if (profile === "wild") {
        stats.cfg     = _nudgeNumeric("cfg", 0.20, 1, 30);
        stats.sampler = _swapCombo("sampler_name", SAMPLER_POOL);
    } else if (profile === "chaos") {
        stats.cfg       = _nudgeNumeric("cfg", 0.50, 1, 30);
        stats.steps     = _nudgeNumeric("steps", 0.50, 4, 80);
        stats.sampler   = _swapCombo("sampler_name", SAMPLER_POOL);
        stats.scheduler = _swapCombo("scheduler", SCHEDULER_POOL);
        stats.spice     = await _maybeSpicePrompt();
    }
    app.canvas?.setDirty?.(true, true);
    return stats;
}

async function _surprise(profile) {
    profile = profile || _defaultProfile();
    const btn = document.getElementById(BTN_ID);
    if (btn) { btn.classList.remove("spinning"); void btn.offsetWidth; btn.classList.add("spinning"); }
    const stats = await _applyProfile(profile);
    console.log("[C2C.SurpriseMe]", stats);
    if (_dryRun()) {
        console.log("[C2C.SurpriseMe] Dry-run: skipped queueing prompt.");
        return;
    }
    try { await app.queuePrompt(0, 1); }
    catch (e) { console.warn("[C2C.SurpriseMe] queue failed:", e); }
}

// ─────────────────────────────────────────────────────────────────────────────
// UI
// ─────────────────────────────────────────────────────────────────────────────

function _updateBadge() {
    const btn = document.getElementById(BTN_ID);
    if (!btn) return;
    const p = _defaultProfile();
    const em = PROFILES[p]?.emoji || "🎰";
    let badge = btn.querySelector(".badge");
    if (!badge) { badge = document.createElement("span"); badge.className = "badge"; btn.appendChild(badge); }
    badge.textContent = em;
}

function _hidePop() {
    const pop = document.getElementById(POP_ID);
    if (pop) pop.classList.remove("visible");
}

function _showPop(x, y) {
    let pop = document.getElementById(POP_ID);
    if (!pop) {
        pop = document.createElement("div");
        pop.id = POP_ID;
        document.body.appendChild(pop);
    }
    const cur = _defaultProfile();
    const rows = Object.entries(PROFILES).map(([k, p]) => `
        <div class="sm-row ${k===cur?"active":""}" data-profile="${k}">
            <span class="sm-emoji">${p.emoji}</span>
            <div style="flex:1;min-width:0;">
                <div class="sm-name">${p.label}</div>
                <div class="sm-desc">${p.desc}</div>
            </div>
        </div>
    `).join("");
    pop.innerHTML = `
        <div class="sm-title">🎰 Surprise Me — pick a profile</div>
        ${rows}
        <hr>
        <label class="sm-opt"><input type="checkbox" data-opt="spice" ${_spiceAllowed()?"checked":""}> AI style-tag (chaos only)</label>
        <label class="sm-opt"><input type="checkbox" data-opt="dry"   ${_dryRun()?"checked":""}> Dry-run (mutate without queue)</label>
        <hr>
        <div class="sm-opt" style="justify-content:flex-end;gap:6px;">
            <button data-act="run" style="background:var(--c2c-pink);border:none;color:var(--c2c-bg);border-radius:4px;padding:4px 10px;cursor:pointer;font-weight:700;font-size:11px;">Run</button>
            <button data-act="cancel" style="background:transparent;border:1px solid var(--c2c-surface0);color:var(--c2c-overlay0);border-radius:4px;padding:4px 10px;cursor:pointer;font-size:11px;">Cancel</button>
        </div>
    `;
    pop.style.left = Math.min(window.innerWidth - 240, x) + "px";
    pop.style.top  = Math.min(window.innerHeight - 240, y) + "px";
    pop.classList.add("visible");

    pop.querySelectorAll(".sm-row").forEach((el) => {
        el.addEventListener("click", () => {
            const k = el.getAttribute("data-profile");
            try { app.ui.settings.setSettingValue("mec.surprise_me.profile", k); } catch {}
            _showPop(x, y);
            _updateBadge();
        });
    });
    pop.querySelector('[data-opt="spice"]').addEventListener("change", (e) => {
        try { app.ui.settings.setSettingValue("mec.surprise_me.spice_prompts", !!e.target.checked); } catch {}
    });
    pop.querySelector('[data-opt="dry"]').addEventListener("change", (e) => {
        try { app.ui.settings.setSettingValue("mec.surprise_me.dry_run", !!e.target.checked); } catch {}
    });
    pop.querySelector('[data-act="run"]').addEventListener("click", () => { _hidePop(); _surprise(_defaultProfile()); });
    pop.querySelector('[data-act="cancel"]').addEventListener("click", _hidePop);
}

function _ensureButton() {
    if (document.getElementById(BTN_ID)) return;
    const b = document.createElement("button");
    b.id = BTN_ID;
    b.title = "Surprise me! (right-click for profiles)";
    b.textContent = "🎰";
    b.addEventListener("click", () => _surprise());
    b.addEventListener("contextmenu", (ev) => {
        ev.preventDefault();
        _showPop(ev.clientX, ev.clientY);
    });
    document.body.appendChild(b);
    _updateBadge();
    // Click-away closes popover
    document.addEventListener("mousedown", (ev) => {
        const pop = document.getElementById(POP_ID);
        if (!pop || !pop.classList.contains("visible")) return;
        if (pop.contains(ev.target) || ev.target === b) return;
        _hidePop();
    });
}

app.registerExtension({
    name: "C2C.SurpriseMe",
    settings: [
        {
            id: "mec.surprise_me.enabled",
            name: "Surprise Me: enabled",
            tooltip: "Show 🎰 button (left-click = use default profile, right-click = picker).",
            type: "boolean",
            default: true,
            onChange: (v) => { const b = document.getElementById(BTN_ID); if (b) b.style.display = v ? "flex" : "none"; },
        },
        {
            id: "mec.surprise_me.profile",
            name: "Surprise Me: default profile",
            tooltip: "Profile used on left-click. Right-click the 🎰 button to switch.",
            type: "combo",
            options: ["mild", "wild", "chaos"],
            default: "mild",
            onChange: () => _updateBadge(),
        },
        {
            id: "mec.surprise_me.spice_prompts",
            name: "Surprise Me: AI style-tag (chaos only)",
            tooltip: "When the chaos profile runs, append one short AI-generated style descriptor to the first positive CLIPTextEncode.",
            type: "boolean",
            default: false,
        },
        {
            id: "mec.surprise_me.dry_run",
            name: "Surprise Me: dry-run (don't queue)",
            tooltip: "Mutate widgets without queueing a prompt — useful for inspecting what changed.",
            type: "boolean",
            default: false,
        },
    ],
    async setup() {
        _injectStyle();
        _ensureButton();
        const b = document.getElementById(BTN_ID);
        if (b) b.style.display = _enabled() ? "flex" : "none";
        console.log("[C2C.SurpriseMe] godlevel-rebuild loaded.");
    },
});
