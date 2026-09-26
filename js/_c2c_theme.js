/**
 * _c2c_theme.js — single source of truth for C2C visual tokens.
 *
 * §0.6 of ideas.md: "One `js/_c2c_theme.js` owns the palette, spacing,
 * radius, z-index scale, motion timings. No panel reimplements the palette
 * inline." This module is that contract.
 *
 * Usage:
 *   import { C, T, applyThemeVars, reducedMotion, z } from "./_c2c_theme.js";
 *
 *   el.style.background = C.bg;
 *   el.style.padding    = T.pad.md;
 *   el.style.zIndex     = z.popover;
 *   el.style.transition = reducedMotion() ? "none" : `opacity ${T.dur.fast}ms ${T.ease.out}`;
 *
 * Or read CSS variables in stylesheets:
 *   color: var(--c2c-fg);
 *   background: var(--c2c-bg);
 *   border-radius: var(--c2c-radius-md);
 *
 * `applyThemeVars()` is called once at module load and re-runs when the
 * user switches variants via the setting `c2c.theme.variant`
 * (night | mocha | oled | latte). §16 theme toggle.
 *
 * `night` is the default and the house identity: the ground is a blue-black
 * indigo ramp and violet is the only saturated hue on it, so the accent reads
 * everywhere without a second colour competing.
 *
 * Backward-compat: `C` is re-exported with the exact same keys as
 * `_c2c_window.js` so existing imports keep working unchanged.
 *
 * License: Apache-2.0
 */

import { app } from "../../scripts/app.js";
// Side-effect import: installs ResizeObserver/MutationObserver that publish
// --c2c-native-top / --c2c-native-left / --c2c-native-bottom on :root so every
// C2C floating surface can anchor itself off ComfyUI's live native chrome
// (workflow tabs, left rail, status strip) instead of guessing fixed pixel
// offsets. Imported here so every consumer of the theme transitively gets it.
import "./_c2c_native_offsets.js";
// Side-effect import: mounts the P0.2 OmniBar (the single, theme-aware,
// position-configurable command bar that hosts ALL future C2C surfaces via
// its window.C2COmniBar.register() slot API). Theme.js is the universally-
// imported entry point, so importing OmniBar here guarantees it boots
// regardless of which C2C extension the user has enabled first.
import "./c2c_omnibar.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";

// ── Palettes ───────────────────────────────────────────────────────────────
// Four variants. Each authors 75 CORE keys by hand - the ground ramp, the
// text tiers, the hue anchors, the handful of semantic shades - and the
// remaining 108 are DERIVED from them at load.
//
// They used to be authored too, and that was the bug. A sweep had replaced
// every inline hex in the codebase with a named token, which is the right
// move, but it pasted the SAME 108 values into all three variants. They were
// byte-identical: 108 of the 183 keys - the majority of the palette - simply
// did not respond to the variant at all. Thirty of them were near-black,
// sitting in `latte`, the LIGHT theme. Any panel reaching for `panelDeep` or
// `scrimDark4` or `neutral900` painted a black block onto a white ground, and
// no amount of care in the individual widgets could fix it, because the
// widgets were doing the right thing.
//
// So the relationships are measured off `mocha` once (see DERIVE below) and
// re-applied to whichever palette is active. Three kinds, because they behave
// differently and conflating them is what produced black-on-white:
//
//   S  surface - a lightness OFFSET from the palette's own ground. Recessed
//      means darker on a dark theme and greyer on a light one; the offset
//      simply flips sign. These must follow the ground or they are wrong.
//   T  tinted background - a surface that carries a hue (an "ok" chip's
//      backing). Ground lightness, family hue.
//   C  chip - foreground colour on the ground. ABSOLUTE lightness, because a
//      bright green should stay bright whether the ground is #1e1e2e or pure
//      black; mirrored about mid on a light theme so it lands dark on white
//      rather than vanishing into it.
//   A  absolute ramp - the greys, same reasoning as C.
//
// Measured against what shipped before: mocha moves 2.6/100 on average (10 of
// 108 shades move enough to see, so the dark theme nobody complained about
// stays put), oled's surfaces finally track its black ground, and latte goes
// from 41 of 41 near-black surfaces to none, with poor-contrast chips down
// from 23 to 2. Adding a palette is now 75 considered colours, not 183.

const CORE = {
    night: {
        bg: "#12132f", bg2: "#0c0d23", bg3: "#07081a", surface0: "#1d1e45",
        surface1: "#2a2a57", surface2: "#3b3a68", overlay0: "#4e4d80", overlay1: "#6664a0",
        overlay2: "#8280ba", fg: "#e8e6f7", sub: "#bab7db", subtext1: "#d2cfee",
        dim: "#6f6d9b", border: "#2a2a57", highlightBase: "#ffffff", shadowBase: "#000000",
        mauve: "#b494ff", blue: "#7ba3f8", sky: "#7fd4f2", sapphire: "#5ec2e3",
        teal: "#76dccb", green: "#8ee09d", yellow: "#f3d288", peach: "#f1aa7b",
        red: "#f27a92", pink: "#eba2de", lavender: "#c6c2ff", rosewater: "#f2e4e2",
        flamingo: "#e7bfbf", maroon: "#e28ca0", panelBg: "#1e1f47", panelBgAlt: "#1d1e45",
        panelTint: "#181a3a", panelHi: "#12132f", panelHi2: "#101128", scrimDark: "#090a1c",
        scrimDark2: "#07081a", accentSoft: "#a892f0", accentSoft2: "#8f74e8", accentLink: "#c0aaff",
        accentLight: "#ddd2ff", accentLight2: "#d4cfe4", accentBright: "#efeaff", accentText: "#e9e4fa",
        accentNeutral: "#e6e5ee", accentMuted: "#8a86a8", accentMuted2: "#847fa4", ok: "#7fe0ab",
        okBright: "#48d97a", okSoft: "#8ee09d", okSoft2: "#6fe3b6", okMute: "#3da874",
        warn: "#ffd27a", warnSoft: "#fde0a0", warnBright: "#f9c93c", warnTint: "#fce9c2",
        danger: "#ff7b93", dangerSoft: "#ff9aad", dangerTint: "#fcb0bd", dangerStrong: "#f87f95",
        violet: "#b9a4e8", violetSoft: "#c68cff", violetTint: "#dcbcff", white: "#ffffff",
        black: "#000000", gray100: "#e2e0ec", gray200: "#cecbdd", gray300: "#aca9c2",
        gray400: "#8b88a3", gray500: "#6a6883", gray600: "#5a5871", gray700: "#48475c",
        gray800: "#37364a", gray900: "#292838", gray950: "#1c1b28",
    },
    mocha: {
        bg: "#1e1e2e", bg2: "#181825", bg3: "#11111b", surface0: "#313244",
        surface1: "#45475a", surface2: "#585b70", overlay0: "#6c7086", overlay1: "#7f849c",
        overlay2: "#9399b2", fg: "#cdd6f4", sub: "#a6adc8", subtext1: "#bac2de",
        dim: "#6c7086", border: "#313244", highlightBase: "#ffffff", shadowBase: "#000000",
        mauve: "#cba6f7", blue: "#89b4fa", sky: "#89dceb", sapphire: "#74c7ec",
        teal: "#94e2d5", green: "#a6e3a1", yellow: "#f9e2af", peach: "#fab387",
        red: "#f38ba8", pink: "#f5c2e7", lavender: "#b4befe", rosewater: "#f5e0dc",
        flamingo: "#f2cdcd", maroon: "#eba0ac", panelBg: "#2a2a36", panelBgAlt: "#2a2a35",
        panelTint: "#22223a", panelHi: "#1a1a2e", panelHi2: "#1a1a26", scrimDark: "#0e0e16",
        scrimDark2: "#0d0d12", accentSoft: "#7bb6f4", accentSoft2: "#5b8def", accentLink: "#9ec1ff",
        accentLight: "#cfe0ff", accentLight2: "#cfd6e0", accentBright: "#e7ecf3", accentText: "#e5ecf5",
        accentNeutral: "#e6e8ec", accentMuted: "#7d8896", accentMuted2: "#7a8492", ok: "#7ee0a8",
        okBright: "#3ecf5a", okSoft: "#a6e3a1", okSoft2: "#6ee7b7", okMute: "#3aa66a",
        warn: "#ffd166", warnSoft: "#fde68a", warnBright: "#facc15", warnTint: "#fce5b6",
        danger: "#ff6b6b", dangerSoft: "#ff8e8e", dangerTint: "#fca5a5", dangerStrong: "#f87171",
        violet: "#b39ddb", violetSoft: "#c084fc", violetTint: "#d8b4fe", white: "#ffffff",
        black: "#000000", gray100: "#e0e0e0", gray200: "#cccccc", gray300: "#aaaaaa",
        gray400: "#888888", gray500: "#666666", gray600: "#555555", gray700: "#444444",
        gray800: "#333333", gray900: "#252525", gray950: "#1a1a1a",
    },
    oled: {
        bg: "#000000", bg2: "#0a0a10", bg3: "#000000", surface0: "#11111b",
        surface1: "#1f2229", surface2: "#2a2a36", overlay0: "#3a3a4a", overlay1: "#5a5f6b",
        overlay2: "#7f849c", fg: "#e8ecf1", sub: "#9aa1ab", subtext1: "#bac2de",
        dim: "#5a5f6b", border: "#1f2229", highlightBase: "#ffffff", shadowBase: "#000000",
        mauve: "#cba6f7", blue: "#89b4fa", sky: "#89dceb", sapphire: "#74c7ec",
        teal: "#94e2d5", green: "#a6e3a1", yellow: "#f9e2af", peach: "#fab387",
        red: "#f38ba8", pink: "#f5c2e7", lavender: "#b4befe", rosewater: "#f5e0dc",
        flamingo: "#f2cdcd", maroon: "#eba0ac", panelBg: "#0a0a10", panelBgAlt: "#0d0d12",
        panelTint: "#0e0e14", panelHi: "#0e0e16", panelHi2: "#0f1218", scrimDark: "#000000",
        scrimDark2: "#000000", accentSoft: "#7bb6f4", accentSoft2: "#5b8def", accentLink: "#9ec1ff",
        accentLight: "#cfe0ff", accentLight2: "#cfd6e0", accentBright: "#e7ecf3", accentText: "#e5ecf5",
        accentNeutral: "#e6e8ec", accentMuted: "#7d8896", accentMuted2: "#7a8492", ok: "#7ee0a8",
        okBright: "#3ecf5a", okSoft: "#a6e3a1", okSoft2: "#6ee7b7", okMute: "#3aa66a",
        warn: "#ffd166", warnSoft: "#fde68a", warnBright: "#facc15", warnTint: "#fce5b6",
        danger: "#ff6b6b", dangerSoft: "#ff8e8e", dangerTint: "#fca5a5", dangerStrong: "#f87171",
        violet: "#b39ddb", violetSoft: "#c084fc", violetTint: "#d8b4fe", white: "#ffffff",
        black: "#000000", gray100: "#e0e0e0", gray200: "#cccccc", gray300: "#aaaaaa",
        gray400: "#888888", gray500: "#666666", gray600: "#555555", gray700: "#444444",
        gray800: "#333333", gray900: "#252525", gray950: "#1a1a1a",
    },
    latte: {
        bg: "#eff1f5", bg2: "#e6e9ef", bg3: "#dce0e8", surface0: "#ccd0da",
        surface1: "#bcc0cc", surface2: "#acb0be", overlay0: "#9ca0b0", overlay1: "#8c8fa1",
        overlay2: "#7c7f93", fg: "#4c4f69", sub: "#5c5f77", subtext1: "#5c5f77",
        dim: "#8c8fa1", border: "#bcc0cc", highlightBase: "#000000", shadowBase: "#000000",
        mauve: "#8839ef", blue: "#1e66f5", sky: "#04a5e5", sapphire: "#209fb5",
        teal: "#179299", green: "#40a02b", yellow: "#df8e1d", peach: "#fe640b",
        red: "#d20f39", pink: "#ea76cb", lavender: "#7287fd", rosewater: "#dc8a78",
        flamingo: "#dd7878", maroon: "#e64553", panelBg: "#e6e9ef", panelBgAlt: "#dce0e8",
        panelTint: "#dce0e8", panelHi: "#eff1f5", panelHi2: "#e6e9ef", scrimDark: "#ccd0da",
        scrimDark2: "#bcc0cc", accentSoft: "#1e66f5", accentSoft2: "#04a5e5", accentLink: "#1e66f5",
        accentLight: "#dce0e8", accentLight2: "#ccd0da", accentBright: "#4c4f69", accentText: "#4c4f69",
        accentNeutral: "#4c4f69", accentMuted: "#6c6f85", accentMuted2: "#7c7f93", ok: "#40a02b",
        okBright: "#40a02b", okSoft: "#40a02b", okSoft2: "#40a02b", okMute: "#40a02b",
        warn: "#df8e1d", warnSoft: "#df8e1d", warnBright: "#df8e1d", warnTint: "#fce5b6",
        danger: "#d20f39", dangerSoft: "#d20f39", dangerTint: "#d20f39", dangerStrong: "#d20f39",
        violet: "#7287fd", violetSoft: "#8839ef", violetTint: "#b4befe", white: "#ffffff",
        black: "#000000", gray100: "#e0e0e0", gray200: "#cccccc", gray300: "#aaaaaa",
        gray400: "#888888", gray500: "#666666", gray600: "#555555", gray700: "#444444",
        gray800: "#333333", gray900: "#252525", gray950: "#1a1a1a",
    },
};

const DERIVE = {accentVivid: ["C", "accent", 0.6804, 1.1975],
    panelDeep: ["S", -0.0529, 0.1429, 0.6786],
    cyanBright: ["C", "cyan", 0.6667, 1.4082],
    dangerBg: ["T", "danger", 0.0235, 0.3182],
    slate400: ["C", "slate", 0.651, 1.5831],
    okBg: ["T", "ok", 0.151, 0.2418],
    panelDeep2: ["S", -0.0392, 0.2143, 1.0179],
    blueDim: ["C", "blue", 0.5961, 0.6446],
    surface1Alt: ["S", 0.1098, 0.1212, 0.5758],
    scrimDark3: ["S", -0.0824, 0.1765, 0.8382],
    panelDeep3: ["S", -0.0451, 0.2453, 1.1651],
    panelDeep4: ["S", -0.0725, 0.2308, 1.0962],
    slate500: ["C", "slate", 0.4608, 0.6995],
    okVivid: ["C", "ok", 0.5804, 1.2784],
    gray350: ["A", 0.6, 0.0],
    neutral900: ["A", 0.1647, 0.0],
    okBg2: ["T", "ok", 0.0843, 0.2437],
    dangerBg2: ["T", "danger", 0.0843, 0.2437],
    dangerBg3: ["T", "danger", 0.151, 0.2418],
    blueSoft: ["C", "blue", 0.8118, 1.0431],
    gray150: ["A", 0.8667, 0.0],
    slateLight: ["C", "slate", 0.651, 1.0554],
    slate300: ["C", "slate", 0.649, 0.8309],
    dangerSoft2: ["C", "danger", 0.7392, 1.2308],
    blueDeep: ["C", "blue", 0.3412, 0.538],
    peachBg: ["T", "peach", 0.1216, 0.3043],
    peachSoft: ["C", "peach", 0.8196, 1.087],
    slate350: ["C", "slate", 0.5961, 0.9879],
    gray120: ["A", 0.902, 0.0],
    neutral850: ["A", 0.2275, 0.0],
    fgAltLight: ["C", "fg", 0.9275, 0.3805],
    cyanMid: ["C", "cyan", 0.6471, 1.1578],
    violetMid: ["C", "violet", 0.6255, 0.6082],
    dangerMid: ["C", "danger", 0.6235, 0.859],
    amberDim: ["C", "amber", 0.5176, 0.3401],
    amberMid: ["C", "amber", 0.5608, 0.716],
    okPale: ["C", "ok", 0.7569, 1.0137],
    neutral950: ["A", 0.1176, 0.0],
    neutral990: ["A", 0.0667, 0.0],
    gray110: ["A", 0.9333, 0.0],
    gray250: ["A", 0.7333, 0.0],
    neutral955: ["A", 0.1098, 0.0],
    scrimDark4: ["S", -0.0784, 0.4444, 2.1111],
    okBgDark: ["T", "ok", 0.0235, 0.3182],
    warnBg: ["T", "warn", 0.0235, 0.3182],
    warnBg2: ["T", "warn", 0.0235, 0.3182],
    okBgDark2: ["T", "ok", -0.0059, 0.3973],
    okBright2: ["C", "ok", 0.6804, 1.1454],
    warnBg3: ["T", "warn", 0.0059, 0.4684],
    amberSoft: ["C", "amber", 0.6667, 0.9571],
    amberStrong: ["C", "amber", 0.5608, 0.8716],
    dangerBgDark: ["T", "danger", 0.0118, 0.4146],
    dangerSoft3: ["C", "danger", 0.7725, 1.2308],
    dangerMid2: ["C", "danger", 0.602, 0.8791],
    panelDeep5: ["S", -0.0549, 0.3333, 1.5833],
    panelDeep6: ["S", 0.0255, 0.3034, 1.441],
    panelDeep7: ["S", -0.0824, 0.4118, 1.9559],
    panelDeep8: ["S", 0.0, 0.3158, 1.5],
    peachMid: ["C", "peach", 0.5392, 0.6984],
    panelDeep9: ["S", 0.0784, 0.2759, 1.3103],
    blueBg: ["T", "blue", 0.0961, 0.52],
    okBgDark3: ["T", "ok", 0.0157, 0.381],
    warnBg4: ["T", "warn", 0.0157, 0.381],
    blueLink: ["C", "blue", 0.3, 1.0885],
    okStrong: ["C", "ok", 0.4863, 1.3416],
    dangerHot: ["C", "danger", 0.6333, 1.2308],
    violetBg: ["T", "violet", 0.1255, 0.2571],
    gray050: ["A", 0.9608, 0.0],
    gray220: ["A", 0.7529, 0.0],
    gray360: ["A", 0.6, 0.0],
    blueAction: ["C", "blue", 0.5059, 0.8898],
    gray450: ["A", 0.4667, 0.0],
    okDeep: ["C", "ok", 0.3353, 0.854],
    panelMid: ["S", 0.1686, 0.284, 1.3488],
    panelMid2: ["S", 0.0588, 0.2075, 0.9858],
    panelMid3: ["S", 0.1059, 0.3538, 1.6808],
    panelMid4: ["S", -0.0039, 0.2973, 1.4122],
    blueSoft2: ["C", "blue", 0.6784, 0.969],
    pinkMid: ["C", "pink", 0.7157, 1.3922],
    okMid: ["C", "ok", 0.6431, 0.711],
    amberSoft2: ["C", "amber", 0.6549, 1.1622],
    cyanSoft: ["C", "cyan", 0.5922, 1.002],
    amberMid2: ["C", "amber", 0.5745, 1.1622],
    okPale2: ["C", "ok", 0.7647, 0.9242],
    violetSoft2: ["C", "violet", 0.7118, 0.5621],
    slate450: ["C", "slate", 0.6235, 1.2231],
    dangerTint2: ["C", "danger", 0.7706, 0.8941],
    tealMid: ["C", "teal", 0.5078, 0.7294],
    dangerSoft4: ["C", "danger", 0.6745, 0.8452],
    panelDeep10: ["S", 0.0157, 0.4762, 2.2619],
    panelDeep11: ["S", -0.0118, 0.1429, 0.6786],
    surface2Alt: ["S", 0.1059, 0.1077, 0.5115],
    peachVivid: ["C", "peach", 0.5, 1.087],
    neutral920: ["A", 0.1333, 0.0],
    okBrightAlt: ["C", "ok", 0.6, 1.2323],
    dangerHotAlt: ["C", "danger", 0.6333, 1.007],
    amberHotAlt: ["C", "amber", 0.7, 1.1622],
    panelTintAlt: ["S", 0.0196, 0.2093, 0.9942],
    scrimDark5: ["S", -0.0706, 0.2, 0.95],
    panelBgAlt2: ["S", 0.0549, 0.1923, 0.9135],
    scrimDark6: ["S", -0.0745, 0.2105, 1.0],
    okMid2: ["C", "ok", 0.649, 0.8984],
    violetBgAlt: ["T", "violet", 0.0176, 0.2],
    neutral910: ["A", 0.1373, 0.0],
    neutral940: ["A", 0.1216, 0.0],
    cyanBright2: ["C", "cyan", 0.7, 1.4082],
    scrimDark7: ["S", -0.0392, 0.1429, 0.6786],
    slateMute: ["C", "slate", 0.5275, 0.6171],
};

// ── Derivation ─────────────────────────────────────────────────────────────
// HSL rather than a perceptual space on purpose: these are UI chrome shades
// read against a known ground, the relationships were measured in the same
// space they are applied in, and it keeps the whole engine dependency-free
// and cheap enough to run on every variant switch.

function _hex2rgb(h) {
    let s = h.replace("#", "");
    if (s.length === 3) s = s.split("").map((c) => c + c).join("");
    return [0, 2, 4].map((i) => parseInt(s.slice(i, i + 2), 16) / 255);
}

function _rgb2hex(c) {
    return "#" + c.map((v) => {
        const n = Math.max(0, Math.min(255, Math.round(v * 255)));
        return n.toString(16).padStart(2, "0");
    }).join("");
}

function _rgb2hsl([r, g, b]) {
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b), l = (mx + mn) / 2;
    if (mx === mn) return [0, 0, l];
    const d = mx - mn;
    const s = l > 0.5 ? d / (2 - mx - mn) : d / (mx + mn);
    let h;
    if (mx === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
    else if (mx === g) h = ((b - r) / d + 2) / 6;
    else h = ((r - g) / d + 4) / 6;
    return [h, s, l];
}

function _hsl2rgb(h, s, l) {
    if (s === 0) return [l, l, l];
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    const f = (t) => {
        t = ((t % 1) + 1) % 1;
        if (t < 1 / 6) return p + (q - p) * 6 * t;
        if (t < 1 / 2) return q;
        if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
        return p;
    };
    return [f(h + 1 / 3), f(h), f(h - 1 / 3)];
}

const _clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));

/**
 * Place a surface at `bl + dl`, folding the offset back when the ground has
 * no room left in that direction.
 *
 * A plain clamp looked fine in the average and destroyed the thing surfaces
 * are FOR. `oled`'s ground is pure black, so every recessed offset hit the
 * floor and seventeen distinct tokens - panelDeep, all five scrims, ten
 * panelDeep variants - collapsed onto #000000. Elevation stopped existing:
 * a panel, the scrim behind a modal and the page all painted the same colour,
 * so nothing had an edge. `latte` had the mirror of it at the top, ten
 * surfaces flattened onto #ffffff.
 *
 * Folding is what a designer does by hand in that situation: on a ground
 * that is already black, "deeper" is drawn as a faintly LIGHTER charcoal,
 * because there is nowhere below. Damped to 0.6 so a fold reads as a quieter
 * step than a real one, and the shades stay distinct from each other.
 */
function _place(bl, dl) {
    const FLOOR = 0.035, CEIL = 0.965;
    let l = bl + dl;
    if (l < FLOOR) l = FLOOR + (FLOOR - l) * 0.6;
    else if (l > CEIL) l = CEIL - (l - CEIL) * 0.6;
    return _clamp(l);
}

/** Which core key each chip/tint family takes its hue from. */
const ANCHOR = {
    ok: "green", danger: "red", warn: "yellow", amber: "yellow", blue: "blue",
    cyan: "sky", violet: "mauve", peach: "peach", pink: "pink", teal: "teal",
    accent: "mauve", slate: "overlay1", fg: "fg",
};

/**
 * Expand a 75-key core into the full 183-key palette.
 * @param {object} core - the hand-authored keys
 * @param {boolean} dark - true if the ground is darker than its text
 * @returns {object} core plus the 108 derived shades
 */
function buildPalette(core, dark) {
    const [, , bl] = _rgb2hsl(_hex2rgb(core.bg));
    // Hue and saturation come from a TONE ANCHOR rather than from bg itself,
    // because `oled`'s ground is pure black: its hue is undefined and its
    // saturation is zero, so anchoring on it would strip the blue out of
    // every surface and leave a flat grey theme. surface1 is the nearest
    // thing to "the ground, but with its colour still attached".
    const raw = _rgb2hsl(_hex2rgb(core.bg));
    const [bh, bs] = raw[1] < 0.05 ? _rgb2hsl(_hex2rgb(core.surface1)) : raw;
    // Only chips flip. Surfaces keep their elevation offset in both themes.
    // On a light theme a shade tuned against #1e1e2e has to land on the other
    // side of mid, or it disappears into the page.
    const mirror = (l) => (dark ? l : _clamp(1 - l));
    const out = { ...core };
    for (const [key, rule] of Object.entries(DERIVE)) {
        const kind = rule[0];
        if (kind === "A") {
            const [, l, s] = rule;
            out[key] = _rgb2hex(_hsl2rgb(bh, Math.min(s + bs * 0.30, 0.34), mirror(l)));
        } else if (kind === "S") {
            // Two numbers, and the shade is their geometric mean.
            //
            // Copying mocha's absolute saturation left night's panels GREY on
            // an indigo ground - the "nothing matches" problem this block
            // exists to fix. Scaling by the palette's own ground saturation
            // instead overshot the other way: night's ground is twice as
            // saturated as mocha's, so panels came out vivid blue-violet,
            // which is not what a panel is for. Half of each keeps the
            // palette's character without letting chrome shout, and reduces
            // to mocha's own value exactly when the palette IS mocha.
            //
            // The offset is NOT mirrored here, unlike a chip. Recessed means
            // darker on a light theme too - that is how elevation reads in
            // both. Mirroring it sent latte's "deep" panels to pure white,
            // because its ground already sits at the top of the range.
            const [, dl, sAbs, sr] = rule;
            const sMix = Math.sqrt(sAbs * _clamp(sr * bs));
            out[key] = _rgb2hex(_hsl2rgb(bh, _clamp(sMix), _place(bl, dl)));
        } else if (kind === "T") {
            // Same reasoning as S for the offset: a tinted backing sits at
            // the ground's elevation, which does not flip with the theme.
            const [, fam, dl, s] = rule;
            const fh = _rgb2hsl(_hex2rgb(core[ANCHOR[fam]]))[0];
            out[key] = _rgb2hex(_hsl2rgb(fh, _clamp(s), _place(bl, dl)));
        } else {
            const [, fam, l, sr] = rule;
            const [ah, as] = _rgb2hsl(_hex2rgb(core[ANCHOR[fam]]));
            out[key] = _rgb2hex(_hsl2rgb(ah, _clamp(as * sr),
                                         _clamp(mirror(l), 0.06, 0.96)));
        }
    }
    return out;
}

/** Which variants read as dark. Drives every sign flip above. */
const IS_DARK = { night: true, mocha: true, oled: true, latte: false };

const PALETTES = Object.fromEntries(
    Object.entries(CORE).map(([name, core]) => [name, buildPalette(core, IS_DARK[name])]),
);

let _variant = "night";
export let C = { ...PALETTES.night };

// Convenience flat-exports: a subset of palette keys imported by name in
// spline_mask_editor.js, spline_mask_tracker.js, and c2c_maskops_health.js.
// They are live `let` bindings so importers see the updated value after
// setVariant() reassigns them below.
// eslint-disable-next-line prefer-const
export let bg3   = C.bg3;
// eslint-disable-next-line prefer-const
export let green = C.green;
// eslint-disable-next-line prefer-const
export let border = C.border;
// eslint-disable-next-line prefer-const
export let peach = C.peach;

// ── Design tokens ──────────────────────────────────────────────────────────
export const T = Object.freeze({
    /** Padding scale (px). */
    pad:    { xs: "2px", sm: "4px", md: "8px", lg: "12px", xl: "16px" },
    /** Border-radius scale (px). */
    radius: { xs: "2px", sm: "4px", md: "6px", lg: "10px", pill: "999px" },
    /** Gap scale (px) — flex/grid. */
    gap:    { xs: "2px", sm: "4px", md: "8px", lg: "12px", xl: "16px" },
    /** Animation durations (ms). */
    dur:    { instant: 0, fast: 120, base: 180, slow: 320 },
    /** Animation easings. */
    ease:   {
        out:    "cubic-bezier(0.16, 1, 0.3, 1)",
        in:     "cubic-bezier(0.5, 0, 0.75, 0)",
        inOut:  "cubic-bezier(0.65, 0, 0.35, 1)",
    },
    /** Elevation shadows. sm=hover chip, md=panel/popover, lg=modal/palette. */
    shadow: {
        sm: "0 1px 2px rgba(0,0,0,0.35)",
        md: "0 6px 18px rgba(0,0,0,0.45)",
        lg: "0 18px 48px rgba(0,0,0,0.55)",
    },
    /** Translucent backdrop overlays (modal scrim, hover wash). */
    overlay: {
        scrim:  "rgba(0,0,0,0.55)",       // full-page modal backdrop
        wash:   "rgba(0,0,0,0.25)",       // light dim
        hover:  "rgba(255,255,255,0.06)", // subtle row-hover tint on dark
    },
    /** Reserved top gutter so overlays never cover ComfyUI's native toolbar. */
    toolbarGutter: 60,
    /** Status bar (top-right host) height. */
    statusBarH:    32,
});

/** Z-index scale — never use a literal number; pick a tier. */
export const z = Object.freeze({
    canvas:       1,      // ComfyUI canvas baseline
    nodeOverlay:  10,     // tiny in-canvas chips bound to a node
    header:       10,     // sticky window headers inside their own panel stacking ctx
    panel:        100,    // sidebar panels, draggable windows
    hud:          1000,   // status bar, complexity HUD, "what's wired"
    dock:         2500,   // shared top-row dock for C2C/MEC overlay buttons
                          //   (sits above HUDs, below ComfyUI/PrimeVue modals)
    popover:      9000,   // tooltips, slot-tip cards, context menus
    modal:        10000,  // settings dialogs, confirm dialogs
    palette:      100001, // Ctrl+K command palette (must beat everything)
    toast:        100002, // C2C error/translator toasts
});

// ── Theme-change subscription ──────────────────────────────────────────────
/**
 * Subscribe to live variant switches. The callback fires AFTER the new
 * palette + CSS vars are committed, so consumers can re-skin DOM that
 * cached colour values at construct time.
 *
 *   const unsub = onThemeChange(({ variant }) => repaint());
 *   // ...later:
 *   unsub();
 *
 * @param {(detail: { variant: string }) => void} cb
 * @returns {() => void} unsubscribe
 */
export function onThemeChange(cb) {
    if (typeof cb !== "function") return () => {};
    const handler = (ev) => {
        try { cb(ev?.detail || { variant: _variant }); }
        catch (err) { console.warn("[c2c-theme] onThemeChange handler threw", err); }
    };
    window.addEventListener("c2c:theme-changed", handler);
    return () => window.removeEventListener("c2c:theme-changed", handler);
}

// ── Reduced motion ─────────────────────────────────────────────────────────
let _reducedMotion = false;
const _mql = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;
function _refreshReducedMotion() {
    _reducedMotion = !!(_mql && _mql.matches);
}
_refreshReducedMotion();
if (_mql) {
    // Older Safari uses addListener; modern uses addEventListener.
    if (_mql.addEventListener) _mql.addEventListener("change", _refreshReducedMotion);
    else if (_mql.addListener)  _mql.addListener(_refreshReducedMotion);
}

/** Returns true if the OS reports prefers-reduced-motion or the user opted in. */
export function reducedMotion() {
    if (_reducedMotion) return true;
    try {
        const v = app?.ui?.settings?.getSettingValue?.("c2c.theme.reducedMotion");
        if (v === true || v === "always") return true;
    } catch (__c2cErr) { __c2cReport("_c2c_theme", __c2cErr); }
    return false;
}

// ── CSS variable injection ─────────────────────────────────────────────────
const STYLE_ID = "c2c-theme-vars";
export function applyThemeVars() {
    let s = document.getElementById(STYLE_ID);
    if (!s) {
        s = document.createElement("style");
        s.id = STYLE_ID;
        document.head.appendChild(s);
    }
    const lines = [":root {"];
    for (const [k, v] of Object.entries(C)) lines.push(`  --c2c-${k}: ${v};`);
    lines.push(`  --c2c-radius-xs: ${T.radius.xs};`);
    lines.push(`  --c2c-radius-sm: ${T.radius.sm};`);
    lines.push(`  --c2c-radius-md: ${T.radius.md};`);
    lines.push(`  --c2c-radius-lg: ${T.radius.lg};`);
    lines.push(`  --c2c-radius-pill: ${T.radius.pill};`);
    lines.push(`  --c2c-pad-xs: ${T.pad.xs};`);
    lines.push(`  --c2c-pad-sm: ${T.pad.sm};`);
    lines.push(`  --c2c-pad-md: ${T.pad.md};`);
    lines.push(`  --c2c-pad-lg: ${T.pad.lg};`);
    lines.push(`  --c2c-pad-xl: ${T.pad.xl};`);
    lines.push(`  --c2c-dur-fast: ${T.dur.fast}ms;`);
    lines.push(`  --c2c-dur-base: ${T.dur.base}ms;`);
    lines.push(`  --c2c-dur-slow: ${T.dur.slow}ms;`);
    lines.push(`  --c2c-ease-out: ${T.ease.out};`);
    lines.push(`  --c2c-ease-in:  ${T.ease.in};`);
    lines.push(`  --c2c-shadow-sm: ${T.shadow.sm};`);
    lines.push(`  --c2c-shadow-md: ${T.shadow.md};`);
    lines.push(`  --c2c-shadow-lg: ${T.shadow.lg};`);
    lines.push(`  --c2c-overlay-scrim: ${T.overlay.scrim};`);
    lines.push(`  --c2c-overlay-wash:  ${T.overlay.wash};`);
    lines.push(`  --c2c-overlay-hover: ${T.overlay.hover};`);
    // Focus ring derived from active palette accent so it tracks variant changes.
    lines.push(`  --c2c-focus-ring: 0 0 0 2px ${C.blue};`);
    // Semantic status aliases — panels should prefer these over raw palette keys.
    lines.push(`  --c2c-status-success: ${C.green};`);
    lines.push(`  --c2c-status-warning: ${C.yellow};`);
    lines.push(`  --c2c-status-danger:  ${C.red};`);
    lines.push(`  --c2c-status-info:    ${C.blue};`);
    // Universal-contrast text color for on-accent buttons. White reads on every
    // saturated mid-tone accent across mocha / latte / oled. Variants may
    // override this in the future if a palette demands different contrast.
    lines.push(`  --c2c-onAccent: #ffffff;`);
    lines.push(`  --c2c-z-panel: ${z.panel};`);
    lines.push(`  --c2c-z-header: ${z.header};`);
    lines.push(`  --c2c-z-hud: ${z.hud};`);
    lines.push(`  --c2c-z-dock: ${z.dock};`);
    lines.push(`  --c2c-z-popover: ${z.popover};`);
    lines.push(`  --c2c-z-modal: ${z.modal};`);
    lines.push(`  --c2c-z-palette: ${z.palette};`);
    lines.push(`  --c2c-z-toast: ${z.toast};`);
    lines.push("}");
    s.textContent = lines.join("\n");
    applyResponsiveCSS();
}

// ── Responsive layer ───────────────────────────────────────────────────────
// Single sheet that retro-fits responsive behavior across every C2C/MEC
// surface without touching the ~40 files that declared fixed pixel widths.
// Rules are scoped to our id/class prefixes so we never collide with native
// ComfyUI components.
//
// Breakpoints (mobile-first, but ComfyUI's native UI assumes desktop):
//   xl   ≥ 1600  full chrome, all labels, full HUDs
//   lg   ≥ 1280  full chrome, all labels
//   md   ≥ 1024  pill labels hidden, HUDs compact
//   sm   ≥ 720   floating HUDs hidden, modals full-bleed
//   xs   < 720   icon-only pills, modals fullscreen-ish
const RESPONSIVE_STYLE_ID = "c2c-responsive-vars";
export function applyResponsiveCSS() {
    let s = document.getElementById(RESPONSIVE_STYLE_ID);
    if (!s) {
        s = document.createElement("style");
        s.id = RESPONSIVE_STYLE_ID;
        document.head.appendChild(s);
    }
    s.textContent = `
/* C2C responsive primitives — consumed by all panels via var(--c2c-*-w/h). */
:root {
    --c2c-bp-xl: 1600px;
    --c2c-bp-lg: 1280px;
    --c2c-bp-md: 1024px;
    --c2c-bp-sm: 720px;
    /* Fluid panel/modal sizing. Panels can opt-in by reading these or by
       falling back to declared px widths (clamped by the universal rules
       below). */
    --c2c-panel-w:  min(540px, calc(100vw - 16px));
    --c2c-popover-w: min(380px, calc(100vw - 16px));
    --c2c-modal-w:  min(720px, calc(100vw - 32px));
    --c2c-modal-w-lg: min(820px, calc(100vw - 32px));
    --c2c-modal-h:  min(720px, calc(100vh - 64px));
    --c2c-gutter:   clamp(8px, 1.2vw, 16px);
}

/* Universal clamp: any C2C fixed/absolute surface must stay inside the
   viewport. Targets only our prefixed surfaces — never native ComfyUI. */
[id^="c2c-"][style*="position: fixed"],
[id^="c2c-"][style*="position:fixed"],
[id^="mec-"][style*="position: fixed"],
[id^="mec-"][style*="position:fixed"],
[id^="c2c-"][style*="position: absolute"],
[id^="mec-"][style*="position: absolute"] {
    max-width:  calc(100vw - 12px);
    max-height: calc(100vh - 24px);
    box-sizing: border-box;
}

/* OmniPill dropdown panel — fluid width, never bleed off-screen. */
#c2c-omnibar {
    width:     var(--c2c-panel-w) !important;
    max-width: var(--c2c-panel-w) !important;
    max-height: calc(100vh - 96px);
    overflow-y: auto;
}

/* OmniPill consolidation (iteration 3) — locked design refinement.
   - OmniPill (Manager bar) = Tools / Bookmarks / AI navigator + Doctor entry.
   - System HUD pill (#c2c-stats-pill) = Crystools-style scroll-cycle of
     live machine state (VRAM / Q / AI $), placement configurable.
   - #mec-complexity-hud STAYS visible at top-center (user explicit request)
     — also publishes to C2CStatusStrip so downstream consumers can render it.
   - Canonical INT badge lives as '#c2c-int-chip' inside the OmniBar; the
     legacy '#mec-integrity-btn' was retired 2026-05-26.
   - Legacy floating #mec-system-hud is hidden — its data flows through
     the registry into the new SysHUD pill. */
#mec-system-hud {
    display: none !important;
}

/* All C2C/MEC modals + popovers honor fluid caps even when declared in px. */
.c2c-modal, .c2c-panel, .c2c-popover,
[class*="c2c-modal"], [class*="c2c-dialog"] {
    max-width: var(--c2c-modal-w);
    max-height: var(--c2c-modal-h);
    box-sizing: border-box;
}

/* Floating HUDs (complexity, system) — keep clear of the canvas and never
   bleed off-screen. */
#mec-complexity-hud, #mec-system-hud {
    max-width: calc(100vw - 16px);
    box-sizing: border-box;
}

/* ── md breakpoint: hide pill text labels, compact HUDs ──────────────── */
@media (max-width: 1280px) {
    /* Icon-only pill row to avoid eating actionbar width. */
    .c2c-omnibar-pill .c2c-omnibar-pill-label,
    #c2c-stats-pill .c2c-sp-lbl,
    #c2c-stats-pill .c2c-sp-cyc { display: none !important; }

    /* HUDs lose their detail trailer, keep the tier badge. */
    #mec-complexity-hud .mec-cx-detail { display: none !important; }
    #mec-complexity-hud { padding: 3px 8px !important; font-size: 10px !important; }
    #mec-system-hud { font-size: 10px !important; padding: 4px 8px !important; }
}

/* ── sm breakpoint: hide floating HUDs, modals fill viewport ─────────── */
@media (max-width: 1024px) {
    /* The complexity meter and system stats live in the OmniBar dropdown
       on small viewports. Hiding the floating chips frees up the canvas. */
    #mec-complexity-hud, #mec-system-hud { display: none !important; }

    .c2c-modal, .c2c-panel,
    [class*="c2c-modal"], [class*="c2c-dialog"] {
        width:  calc(100vw - 24px) !important;
        max-width: calc(100vw - 24px) !important;
        max-height: calc(100vh - 48px);
    }

    /* OmniBar panel becomes near-full-bleed so usable on tablets. */
    #c2c-omnibar {
        width:     calc(100vw - 24px) !important;
        max-width: calc(100vw - 24px) !important;
    }
}

/* ── xs breakpoint: very narrow — keep core actions, hide chrome ─────── */
@media (max-width: 720px) {
    /* Stats pill is decorative without labels; drop it entirely. */
    #c2c-stats-pill { display: none !important; }
}

/* Respect user's reduced-motion preference for any C2C transitions. */
@media (prefers-reduced-motion: reduce) {
    [id^="c2c-"], [id^="mec-"], [class*="c2c-"], [class*="mec-"] {
        animation-duration: 0.001ms !important;
        transition-duration: 0.001ms !important;
    }
}

/* ── OmniPill <-> Stats-pill alignment (sit on same baseline) ─────────── */
/* Both pills must share identical box-sizing + height + vertical-alignment
   so they line up perfectly in the native Manager bar. Force margins,
   border-box, and middle alignment in case any ancestor flex container
   tries to stretch them. */
#c2c-omnibar-pill,
#c2c-stats-pill {
    box-sizing: border-box !important;
    height: 28px !important;
    margin: 0 3px !important;
    vertical-align: middle !important;
    align-self: center !important;
}

/* ── Window-chrome shortcut hint (panel titles) ───────────────────────── */
/* "(Ctrl+Shift+W)" style hint embedded in every C2C panel title. Same
   typographic weight across Inspector / What's Wired / OmniPill panels. */
.c2c-win-shortcut,
.ww-shortcut,
.c2c-ne-panel-shortcut {
    color: var(--c2c-sub, #9399b2);
    font-weight: 400;
    font-size: 10px;
    letter-spacing: 0.02em;
    margin-left: 6px;
    opacity: 0.8;
}

/* ── Node-body multiline-string widget polish ─────────────────────────── */
/* ComfyUI ships .comfy-multiline-input (the JSON / prompt / customtext
   textarea inside a node) with a stark near-black background and no
   border. The DOM textarea sits on top of a LiteGraph CANVAS that paints
   its own solid-black widget rectangle underneath — so a translucent
   textarea bg lets that black slab bleed through and looks like an
   unmotivated "black box" on the node.

   CRITICAL: the background MUST be opaque, otherwise the canvas-painted
   black widget shape shows through (verified live 2026-05-28). We use a
   solid panel shade slightly lighter than the typical ComfyUI node body
   (#353535) so the textarea is visually delimited but not jarring. */
textarea.comfy-multiline-input {
    background: #2b2b30 !important;
    color: var(--c2c-fg, #e6e6e6) !important;
    border: 1px solid rgba(255, 255, 255, 0.10) !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
    box-sizing: border-box !important;
    transition: border-color var(--c2c-dur-fast, 120ms) var(--c2c-ease-out, ease-out),
                background     var(--c2c-dur-fast, 120ms) var(--c2c-ease-out, ease-out);
}
textarea.comfy-multiline-input:hover {
    background: #32323a !important;
    border-color: rgba(255, 255, 255, 0.18) !important;
}
textarea.comfy-multiline-input:focus {
    background: #32323a !important;
    outline: none !important;
    border-color: var(--c2c-blue, #89b4fa) !important;
    box-shadow: 0 0 0 2px rgba(137, 180, 250, 0.35) !important;
}
`;
}

/** Switch active variant. Triggers a full CSS-var reflow. */
export function setVariant(name) {
    const v = String(name || "").toLowerCase();
    if (!PALETTES[v]) return false;
    _variant = v;
    Object.assign(C, PALETTES[v]);
    // Keep the flat-binding convenience exports in sync.
    bg3 = C.bg3; green = C.green; border = C.border; peach = C.peach;
    applyThemeVars();
    try {
        window.dispatchEvent(new CustomEvent("c2c:theme-changed", { detail: { variant: v } }));
    } catch (dispatchErr) {
        // Theme dispatch failure is non-fatal (variant DID apply via applyThemeVars()
        // above) but it MUST be surfaced — listeners (OmniBar, status strip, etc.)
        // expect this event to repaint. Route through the registry-failure channel
        // so the user sees it instead of a silent swallow.
        // eslint-disable-next-line no-console
        console.error("[c2c_theme] c2c:theme-changed dispatch failed", dispatchErr);
        try {
            window.dispatchEvent(new CustomEvent("c2c:registry-failure", {
                detail: {
                    component: "c2c_theme",
                    where: "setVariant:dispatch",
                    message: dispatchErr && dispatchErr.message ? dispatchErr.message : String(dispatchErr),
                    stack: dispatchErr && dispatchErr.stack ? dispatchErr.stack : null,
                    ts: Date.now(),
                },
            }));
        } catch (innerErr) {
            // eslint-disable-next-line no-console
            console.error("[c2c_theme] failed to dispatch registry-failure", innerErr);
        }
    }
    return true;
}
export function getVariant() { return _variant; }
export function listVariants() { return Object.keys(PALETTES); }

// ── First-load wiring ──────────────────────────────────────────────────────
applyThemeVars();

// Register a setting so the user can switch variants from ComfyUI's Settings.
if (!window.__C2C_THEME_REG__) { window.__C2C_THEME_REG__ = true; try {
    app.registerExtension({
        name: "C2C.Theme",
        settings: [
            {
                id: "c2c.theme.variant",
                name: "C2C → Theme variant",
                tooltip: "Palette used across every C2C panel, HUD and widget.",
                type: "combo",
                options: [
                    { value: "night", text: "Night (default) — indigo ground, violet accent" },
                    { value: "mocha", text: "Mocha (Catppuccin dark)" },
                    { value: "oled",  text: "OLED (true black)" },
                    { value: "latte", text: "Latte (light)" },
                ],
                defaultValue: "night",
                onChange: (v) => setVariant(v),
            },
            {
                id: "c2c.theme.reducedMotion",
                name: "C2C → Force reduced motion",
                tooltip: "Suppress non-essential animations regardless of OS setting.",
                type: "boolean",
                defaultValue: false,
            },
        ],
    });
} catch (__c2cErr) { __c2cReport("_c2c_theme", __c2cErr); } }
