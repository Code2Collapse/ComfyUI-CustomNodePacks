/**
 * mec_noodle_styles.js — custom link rendering styles for ComfyUI.
 *
 * Patches `LGraphCanvas.prototype.renderLink` with a dispatcher that
 * looks at the `mec.noodle.style` setting and either calls the original
 * renderer (for `default`) or one of the styles defined below.
 *
 * Adds a "🍝 Noodle Style →" submenu to the canvas right-click menu.
 *
 * Styles:
 *   default       — ComfyUI built-in (unmodified).
 *   spider-web    — Tobey 🕷️ sprite rides the midpoint of every link.
 *   lightsaber    — white core + colored glow, type-keyed:
 *                       MODEL→red, IMAGE→blue, MASK→green, default→purple.
 *   dna-helix     — two phase-shifted sines, link-type tinted.
 *   rainbow-flow  — animated HSL gradient flowing output→input.
 *   dashed-march  — marching ants (setLineDash + lineDashOffset tick).
 *   lightning     — perturbed polyline, redraws every 3rd frame.
 *   pulse-packet  — bezier + a moving "data packet" dot.
 *   neon-tube     — bright thin core + dim wide glow (no animation).
 *
 * The setting key `mec.noodle.style` is also exposed in the ComfyUI
 * settings panel so the user can change it without using the menu.
 */

import { app } from "../../scripts/app.js";
import { C } from './_c2c_theme.js';
import { reportFailure as __c2cReport } from "./_c2c_report.js";

const STYLES = [
    "default","spider-web","lightsaber","dna-helix","rainbow-flow",
    "dashed-march","lightning","pulse-packet","neon-tube",
];
const SETTING_ID = "mec.noodle.style";

let _orig = null;

// Catppuccin-keyed default link colors per type.
function _typeColor(linkType) {
    const t = (linkType || "").toUpperCase();
    if (t.includes("MODEL"))      return "var(--c2c-red)";      // red
    if (t.includes("IMAGE"))      return "var(--c2c-blue)";      // blue
    if (t.includes("MASK"))       return "var(--c2c-green)";      // green
    if (t.includes("LATENT"))     return "var(--c2c-mauve)";      // mauve
    if (t.includes("CLIP"))       return "var(--c2c-yellow)";      // yellow
    if (t.includes("VAE"))        return "var(--c2c-peach)";      // peach
    if (t.includes("CONDITION"))  return "var(--c2c-teal)";      // teal
    return "var(--c2c-lavender)";                                    // lavender
}

function _bezierPoints(a, b) {
    const dist = Math.max(20, Math.abs(a[0] - b[0]) * 0.5);
    return [[a[0]+dist, a[1]], [b[0]-dist, b[1]]];
}
function _bezierAt(t, a, cp1, cp2, b) {
    const u = 1 - t;
    const x = u*u*u*a[0] + 3*u*u*t*cp1[0] + 3*u*t*t*cp2[0] + t*t*t*b[0];
    const y = u*u*u*a[1] + 3*u*u*t*cp1[1] + 3*u*t*t*cp2[1] + t*t*t*b[1];
    return [x, y];
}

// ── Style implementations ────────────────────────────────────────────
function _renderSpiderWeb(ctx, a, b, color) {
    const [cp1, cp2] = _bezierPoints(a, b);
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(a[0],a[1]);
    ctx.bezierCurveTo(cp1[0],cp1[1], cp2[0],cp2[1], b[0],b[1]); ctx.stroke();
    const [mx, my] = _bezierAt(0.5, a, cp1, cp2, b);
    ctx.font = "18px serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText("🕷️", mx, my);
    ctx.restore();
}
function _renderLightsaber(ctx, a, b, color, linkType) {
    const [cp1, cp2] = _bezierPoints(a, b);
    const glow = _typeColor(linkType);
    ctx.save();
    // Outer glow
    ctx.strokeStyle = glow; ctx.lineWidth = 8; ctx.globalAlpha = 0.45;
    ctx.shadowBlur = 16; ctx.shadowColor = glow;
    ctx.beginPath(); ctx.moveTo(a[0],a[1]);
    ctx.bezierCurveTo(cp1[0],cp1[1], cp2[0],cp2[1], b[0],b[1]); ctx.stroke();
    // White core
    ctx.globalAlpha = 1; ctx.shadowBlur = 0;
    ctx.strokeStyle = C.white; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(a[0],a[1]);
    ctx.bezierCurveTo(cp1[0],cp1[1], cp2[0],cp2[1], b[0],b[1]); ctx.stroke();
    ctx.restore();
}
function _renderDnaHelix(ctx, a, b, color) {
    const [cp1, cp2] = _bezierPoints(a, b);
    const N = 40;
    ctx.save();
    const phase = (performance.now()/300) % (Math.PI*2);
    for (let strand=0; strand<2; strand++) {
        ctx.strokeStyle = strand===0 ? color : _typeColor("");
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let i=0;i<=N;i++) {
            const t = i/N;
            const [bx,by] = _bezierAt(t, a, cp1, cp2, b);
            // perpendicular offset
            const [bx2,by2] = _bezierAt(Math.min(1,t+0.01), a, cp1, cp2, b);
            const dx = bx2-bx, dy = by2-by, len = Math.hypot(dx,dy)||1;
            const nx = -dy/len, ny = dx/len;
            const off = Math.sin(t*Math.PI*6 + phase + strand*Math.PI) * 8;
            const px = bx + nx*off, py = by + ny*off;
            if (i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
        }
        ctx.stroke();
    }
    ctx.restore();
}
function _renderRainbowFlow(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    const t = performance.now()/30;
    const grad = ctx.createLinearGradient(a[0],a[1],b[0],b[1]);
    for (let i=0;i<=6;i++) {
        const hue = ((i*60)+t) % 360;
        grad.addColorStop(i/6, `hsl(${hue}, 80%, 65%)`);
    }
    ctx.save();
    ctx.strokeStyle = grad; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(a[0],a[1]);
    ctx.bezierCurveTo(cp1[0],cp1[1], cp2[0],cp2[1], b[0],b[1]); ctx.stroke();
    ctx.restore();
}
function _renderDashedMarch(ctx, a, b, color) {
    const [cp1, cp2] = _bezierPoints(a, b);
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.setLineDash([8,6]);
    ctx.lineDashOffset = -((performance.now()/40) % 14);
    ctx.beginPath(); ctx.moveTo(a[0],a[1]);
    ctx.bezierCurveTo(cp1[0],cp1[1], cp2[0],cp2[1], b[0],b[1]); ctx.stroke();
    ctx.setLineDash([]); ctx.restore();
}
function _renderLightning(ctx, a, b, color) {
    const [cp1, cp2] = _bezierPoints(a, b);
    const N = 10;
    // Only re-jitter every 3rd frame to avoid epilepsy + perf hit.
    const phase = Math.floor(performance.now()/100);
    const rand = (i) => {
        const s = Math.sin(i*12.9898 + phase*78.233) * 43758.5453;
        return s - Math.floor(s);
    };
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = 1.5;
    ctx.shadowBlur = 8; ctx.shadowColor = color;
    ctx.beginPath();
    for (let i=0;i<=N;i++) {
        const t = i/N;
        const [bx,by] = _bezierAt(t, a, cp1, cp2, b);
        const jx = (rand(i)-0.5)*16, jy = (rand(i+100)-0.5)*16;
        if (i===0) ctx.moveTo(bx,by); else ctx.lineTo(bx+jx, by+jy);
    }
    ctx.stroke(); ctx.restore();
}
function _renderPulsePacket(ctx, a, b, color) {
    const [cp1, cp2] = _bezierPoints(a, b);
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.globalAlpha = 0.7;
    ctx.beginPath(); ctx.moveTo(a[0],a[1]);
    ctx.bezierCurveTo(cp1[0],cp1[1], cp2[0],cp2[1], b[0],b[1]); ctx.stroke();
    ctx.globalAlpha = 1;
    const t = ((performance.now()/1500) % 1);
    const [px, py] = _bezierAt(t, a, cp1, cp2, b);
    ctx.fillStyle = C.white; ctx.shadowBlur = 12; ctx.shadowColor = color;
    ctx.beginPath(); ctx.arc(px,py,4,0,Math.PI*2); ctx.fill();
    ctx.restore();
}
function _renderNeonTube(ctx, a, b, color) {
    const [cp1, cp2] = _bezierPoints(a, b);
    ctx.save();
    // wide dim glow
    ctx.strokeStyle = color; ctx.lineWidth = 6; ctx.globalAlpha = 0.25;
    ctx.shadowBlur = 18; ctx.shadowColor = color;
    ctx.beginPath(); ctx.moveTo(a[0],a[1]);
    ctx.bezierCurveTo(cp1[0],cp1[1], cp2[0],cp2[1], b[0],b[1]); ctx.stroke();
    // bright thin core
    ctx.globalAlpha = 1; ctx.shadowBlur = 0;
    ctx.strokeStyle = C.white; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(a[0],a[1]);
    ctx.bezierCurveTo(cp1[0],cp1[1], cp2[0],cp2[1], b[0],b[1]); ctx.stroke();
    ctx.restore();
}

const _RENDER = {
    "spider-web":   _renderSpiderWeb,
    "lightsaber":   _renderLightsaber,
    "dna-helix":    _renderDnaHelix,
    "rainbow-flow": _renderRainbowFlow,
    "dashed-march": _renderDashedMarch,
    "lightning":    _renderLightning,
    "pulse-packet": _renderPulsePacket,
    "neon-tube":    _renderNeonTube,
};

function _currentStyle() {
    try { return app.ui.settings.getSettingValue(SETTING_ID, "default"); }
    catch (_) { return "default"; }
}

function _installRenderPatch() {
    if (_orig || !window.LGraphCanvas) return;
    _orig = LGraphCanvas.prototype.renderLink;
    LGraphCanvas.prototype.renderLink = function (ctx, a, b, link, skip_border, flow, color, start_dir, end_dir, num_sublines) {
        const style = _currentStyle();
        if (style === "default" || !_RENDER[style]) {
            return _orig.call(this, ctx, a, b, link, skip_border, flow, color, start_dir, end_dir, num_sublines);
        }
        try {
            const linkType = link?.type || "";
            const effColor = color || _typeColor(linkType);
            _RENDER[style](ctx, a, b, effColor, linkType);
            // Force a continuous redraw for animated styles
            if (["dna-helix","rainbow-flow","dashed-march","lightning","pulse-packet"].includes(style)) {
                this.dirty_canvas = true;
            }
        } catch (e) {
            console.warn("[MEC.NoodleStyles] render error, falling back:", e);
            return _orig.call(this, ctx, a, b, link, skip_border, flow, color, start_dir, end_dir, num_sublines);
        }
    };
}

function _installMenuPatch() {
    if (!window.LGraphCanvas) return;
    const origMenu = LGraphCanvas.prototype.getCanvasMenuOptions;
    LGraphCanvas.prototype.getCanvasMenuOptions = function () {
        const out = origMenu ? origMenu.call(this) : [];
        const cur = _currentStyle();
        const submenu = STYLES.map(s => ({
            content: `${s === cur ? "✓ " : "  "}${s}`,
            callback: () => {
                try { app.ui.settings.setSettingValue(SETTING_ID, s); } catch (__c2cErr) { __c2cReport("c2c_noodle_styles", __c2cErr); }
                this.setDirty(true, true);
            },
        }));
        out.push(null); // separator
        out.push({
            content: "🍝 Noodle Style",
            has_submenu: true,
            submenu: { options: submenu },
        });
        out.push({
            content: "— also configurable in Settings → mec.noodle.style",
            disabled: true,
        });
        return out;
    };
}

app.registerExtension({
    name: "C2C.NoodleStyles",
    settings: [
        {
            id: SETTING_ID,
            name: "Noodle style",
            tooltip:
                "Custom rendering for node links. `default` keeps ComfyUI's built-in. " +
                "Also reachable from the canvas right-click menu (🍝 Noodle Style) and " +
                "from the reroute-dot context menu.",
            type: "combo",
            options: STYLES,
            defaultValue: "default",
        },
    ],
    async setup() {
        // LiteGraph is loaded before extensions, but be defensive.
        const tryInstall = () => {
            if (window.LGraphCanvas) {
                _installRenderPatch();
                _installMenuPatch();
                console.log("[MEC.NoodleStyles] Installed:", STYLES.filter(s=>s!=="default").join(", "));
            } else {
                setTimeout(tryInstall, 100);
            }
        };
        tryInstall();
    },
});

// Expose to other modules (e.g. universal_reroute.js menu mirror).
export const NOODLE_STYLES = STYLES;
export const NOODLE_SETTING_ID = SETTING_ID;
