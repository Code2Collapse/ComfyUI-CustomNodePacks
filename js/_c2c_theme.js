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
 * user switches Catppuccin variants via the setting `c2c.theme.variant`
 * (mocha | latte | oled | custom). §16 theme toggle.
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

// ── Catppuccin variants ────────────────────────────────────────────────────
// Palette keys with extended Catppuccin surface/overlay/subtext tiers + a few
// long-standing C2C semantic shades (panelBg, panelBgAlt, scrim*, accent*,
// danger*, warn*, ok*) so every literal hex in the codebase has a named token
// to resolve against. P0.1b theme-sweep relies on this expanded surface set.
const PALETTES = {
    mocha: {
        // base layers
        bg: "#1e1e2e", bg2: "#181825", bg3: "#11111b",
        // surface tiers (Catppuccin surface0/1/2)
        surface0: "#313244", surface1: "#45475a", surface2: "#585b70",
        // overlay tiers (Catppuccin overlay0/1/2)
        overlay0: "#6c7086", overlay1: "#7f849c", overlay2: "#9399b2",
        // foreground tiers (text + subtext0/1)
        fg: "#cdd6f4", sub: "#a6adc8", subtext1: "#bac2de", dim: "#6c7086",
        // generic alias kept for back-compat
        border: "#313244",
        // Per-variant translucency anchors. shadowBase is universally black;
        // highlightBase flips to keep "light wash" overlays visible in latte.
        highlightBase: "#ffffff", shadowBase: "#000000",
        // accents
        mauve: "#cba6f7", blue: "#89b4fa", sky: "#89dceb", sapphire: "#74c7ec",
        teal: "#94e2d5", green: "#a6e3a1", yellow: "#f9e2af",
        peach: "#fab387", red: "#f38ba8", pink: "#f5c2e7",
        lavender: "#b4befe", rosewater: "#f5e0dc", flamingo: "#f2cdcd", maroon: "#eba0ac",
        // C2C-specific named shades long used inline across panels
        panelBg: "#2a2a36", panelBgAlt: "#2a2a35",
        panelTint: "#22223a", panelHi: "#1a1a2e", panelHi2: "#1a1a26",
        scrimDark: "#0e0e16", scrimDark2: "#0d0d12",
        accentSoft: "#7bb6f4", accentSoft2: "#5b8def", accentLink: "#9ec1ff",
        accentLight: "#cfe0ff", accentLight2: "#cfd6e0",
        accentBright: "#e7ecf3", accentText: "#e5ecf5", accentNeutral: "#e6e8ec",
        accentMuted: "#7d8896", accentMuted2: "#7a8492",
        ok: "#7ee0a8", okBright: "#3ecf5a", okSoft: "#a6e3a1", okSoft2: "#6ee7b7", okMute: "#3aa66a",
        warn: "#ffd166", warnSoft: "#fde68a", warnBright: "#facc15", warnTint: "#fce5b6",
        danger: "#ff6b6b", dangerSoft: "#ff8e8e", dangerTint: "#fca5a5", dangerStrong: "#f87171",
        violet: "#b39ddb", violetSoft: "#c084fc", violetTint: "#d8b4fe",
        // common neutrals (CSS shorthand collapsed to canonical 6-char hex)
        white: "#ffffff", black: "#000000", gray100: "#e0e0e0", gray200: "#cccccc",
        gray300: "#aaaaaa", gray400: "#888888", gray500: "#666666", gray600: "#555555",
        gray700: "#444444", gray800: "#333333", gray900: "#252525", gray950: "#1a1a1a",
        // __P0_1B_EXTRAS__
        accentVivid: "#7c5cff", panelDeep: "#15151c", cyanBright: "#55bbff", dangerBg: "#3a1e29",
        slate400: "#94a3b8", okBg: "#3a5f50", panelDeep2: "#161a22", blueDim: "#5b9bd5",
        surface1Alt: "#3a3a4a", scrimDark3: "#0e0e14", panelDeep3: "#141821", panelDeep4: "#0f1218",
        slate500: "#6b7280", okVivid: "#4ade80", gray350: "#999999", neutral900: "#2a2a2a",
        okBg2: "#2d4a3e", dangerBg2: "#4a2d2d", dangerBg3: "#5f3a3a", blueSoft: "#a1c4fd",
        gray150: "#dddddd", slateLight: "#9aa6b2", slate300: "#9ca3af", dangerSoft2: "#ff7a7a",
        blueDeep: "#2c4a82", peachBg: "#5a3a30", peachSoft: "#ffd1a3", slate350: "#8b96a5",
        gray120: "#e6e6e6", neutral850: "#3a3a3a", fgAltLight: "#e8ecf1", cyanMid: "#5bd3ef",
        violetMid: "#a06fd0", dangerMid: "#e25c5c", amberDim: "#a89060", amberMid: "#d4a04a",
        okPale: "#9fe39f", neutral950: "#1e1e1e", neutral990: "#111111", gray110: "#eeeeee",
        gray250: "#bbbbbb", neutral955: "#1c1c1c", scrimDark4: "#0a0a1a", okBgDark: "#1e3a2a",
        warnBg: "#3a361e", warnBg2: "#3a311e", okBgDark2: "#16331f", okBright2: "#7be089",
        warnBg3: "#3a2e15", amberSoft: "#f0c764", amberStrong: "#e3a93b", dangerBgDark: "#3a1818",
        dangerSoft3: "#ff8b8b", dangerMid2: "#e25151", panelDeep5: "#101820", panelDeep6: "#1f2c3a",
        panelDeep7: "#0a1218", panelDeep8: "#1a2632", peachMid: "#d58a3e", panelDeep9: "#2a3a4a",
        blueBg: "#1e3a5f", okBgDark3: "#1a3a1a", warnBg4: "#3a2a1a", blueLink: "#003399",
        okStrong: "#22d65a", dangerHot: "#ff4466", violetBg: "#583444", gray050: "#f5f5f5",
        gray220: "#c0c0c0", gray360: "#999999", blueAction: "#1a73e8", gray450: "#777777",
        okDeep: "#2e7d32", panelMid: "#3a5068", panelMid2: "#2a3040", panelMid3: "#2a4058",
        panelMid4: "#1a2030", blueSoft2: "#64b5f6", pinkMid: "#ff6e9c", okMid: "#81c784",
        amberSoft2: "#ffd54f", cyanSoft: "#4dd0e1", amberMid2: "#ffa726", okPale2: "#c5e1a5",
        violetSoft2: "#ce93d8", slate450: "#90a4ae", dangerTint2: "#ef9a9a", tealMid: "#4db6ac",
        dangerSoft4: "#e57373", panelDeep10: "#16213e", panelDeep11: "#1e1e28", surface2Alt: "#3a3a48",
        peachVivid: "#ff8800", neutral920: "#222222", okBrightAlt: "#55dd55", dangerHotAlt: "#ee5555",
        amberHotAlt: "#ffcc66", panelTintAlt: "#222234", scrimDark5: "#101018", panelBgAlt2: "#2a2a3e",
        scrimDark6: "#0f0f17", okMid2: "#7fd17a", violetBgAlt: "#332233", neutral910: "#232323",
        neutral940: "#1f1f1f", cyanBright2: "#66ccff", scrimDark7: "#181820", slateMute: "#7d8590",
    },
    latte: {
        bg: "#eff1f5", bg2: "#e6e9ef", bg3: "#dce0e8",
        surface0: "#ccd0da", surface1: "#bcc0cc", surface2: "#acb0be",
        overlay0: "#9ca0b0", overlay1: "#8c8fa1", overlay2: "#7c7f93",
        fg: "#4c4f69", sub: "#5c5f77", subtext1: "#5c5f77", dim: "#8c8fa1",
        border: "#bcc0cc",
        // Light variant: highlightBase flips to dark so overlays remain visible.
        highlightBase: "#000000", shadowBase: "#000000",
        mauve: "#8839ef", blue: "#1e66f5", sky: "#04a5e5", sapphire: "#209fb5",
        teal: "#179299", green: "#40a02b", yellow: "#df8e1d",
        peach: "#fe640b", red: "#d20f39", pink: "#ea76cb",
        lavender: "#7287fd", rosewater: "#dc8a78", flamingo: "#dd7878", maroon: "#e64553",
        panelBg: "#e6e9ef", panelBgAlt: "#dce0e8",
        panelTint: "#dce0e8", panelHi: "#eff1f5", panelHi2: "#e6e9ef",
        scrimDark: "#ccd0da", scrimDark2: "#bcc0cc",
        accentSoft: "#1e66f5", accentSoft2: "#04a5e5", accentLink: "#1e66f5",
        accentLight: "#dce0e8", accentLight2: "#ccd0da",
        accentBright: "#4c4f69", accentText: "#4c4f69", accentNeutral: "#4c4f69",
        accentMuted: "#6c6f85", accentMuted2: "#7c7f93",
        ok: "#40a02b", okBright: "#40a02b", okSoft: "#40a02b", okSoft2: "#40a02b", okMute: "#40a02b",
        warn: "#df8e1d", warnSoft: "#df8e1d", warnBright: "#df8e1d", warnTint: "#fce5b6",
        danger: "#d20f39", dangerSoft: "#d20f39", dangerTint: "#d20f39", dangerStrong: "#d20f39",
        violet: "#7287fd", violetSoft: "#8839ef", violetTint: "#b4befe",
        white: "#ffffff", black: "#000000", gray100: "#e0e0e0", gray200: "#cccccc",
        gray300: "#aaaaaa", gray400: "#888888", gray500: "#666666", gray600: "#555555",
        gray700: "#444444", gray800: "#333333", gray900: "#252525", gray950: "#1a1a1a",
        // __P0_1B_EXTRAS__
        accentVivid: "#7c5cff", panelDeep: "#15151c", cyanBright: "#55bbff", dangerBg: "#3a1e29",
        slate400: "#94a3b8", okBg: "#3a5f50", panelDeep2: "#161a22", blueDim: "#5b9bd5",
        surface1Alt: "#3a3a4a", scrimDark3: "#0e0e14", panelDeep3: "#141821", panelDeep4: "#0f1218",
        slate500: "#6b7280", okVivid: "#4ade80", gray350: "#999999", neutral900: "#2a2a2a",
        okBg2: "#2d4a3e", dangerBg2: "#4a2d2d", dangerBg3: "#5f3a3a", blueSoft: "#a1c4fd",
        gray150: "#dddddd", slateLight: "#9aa6b2", slate300: "#9ca3af", dangerSoft2: "#ff7a7a",
        blueDeep: "#2c4a82", peachBg: "#5a3a30", peachSoft: "#ffd1a3", slate350: "#8b96a5",
        gray120: "#e6e6e6", neutral850: "#3a3a3a", fgAltLight: "#e8ecf1", cyanMid: "#5bd3ef",
        violetMid: "#a06fd0", dangerMid: "#e25c5c", amberDim: "#a89060", amberMid: "#d4a04a",
        okPale: "#9fe39f", neutral950: "#1e1e1e", neutral990: "#111111", gray110: "#eeeeee",
        gray250: "#bbbbbb", neutral955: "#1c1c1c", scrimDark4: "#0a0a1a", okBgDark: "#1e3a2a",
        warnBg: "#3a361e", warnBg2: "#3a311e", okBgDark2: "#16331f", okBright2: "#7be089",
        warnBg3: "#3a2e15", amberSoft: "#f0c764", amberStrong: "#e3a93b", dangerBgDark: "#3a1818",
        dangerSoft3: "#ff8b8b", dangerMid2: "#e25151", panelDeep5: "#101820", panelDeep6: "#1f2c3a",
        panelDeep7: "#0a1218", panelDeep8: "#1a2632", peachMid: "#d58a3e", panelDeep9: "#2a3a4a",
        blueBg: "#1e3a5f", okBgDark3: "#1a3a1a", warnBg4: "#3a2a1a", blueLink: "#003399",
        okStrong: "#22d65a", dangerHot: "#ff4466", violetBg: "#583444", gray050: "#f5f5f5",
        gray220: "#c0c0c0", gray360: "#999999", blueAction: "#1a73e8", gray450: "#777777",
        okDeep: "#2e7d32", panelMid: "#3a5068", panelMid2: "#2a3040", panelMid3: "#2a4058",
        panelMid4: "#1a2030", blueSoft2: "#64b5f6", pinkMid: "#ff6e9c", okMid: "#81c784",
        amberSoft2: "#ffd54f", cyanSoft: "#4dd0e1", amberMid2: "#ffa726", okPale2: "#c5e1a5",
        violetSoft2: "#ce93d8", slate450: "#90a4ae", dangerTint2: "#ef9a9a", tealMid: "#4db6ac",
        dangerSoft4: "#e57373", panelDeep10: "#16213e", panelDeep11: "#1e1e28", surface2Alt: "#3a3a48",
        peachVivid: "#ff8800", neutral920: "#222222", okBrightAlt: "#55dd55", dangerHotAlt: "#ee5555",
        amberHotAlt: "#ffcc66", panelTintAlt: "#222234", scrimDark5: "#101018", panelBgAlt2: "#2a2a3e",
        scrimDark6: "#0f0f17", okMid2: "#7fd17a", violetBgAlt: "#332233", neutral910: "#232323",
        neutral940: "#1f1f1f", cyanBright2: "#66ccff", scrimDark7: "#181820", slateMute: "#7d8590",
    },
    oled: {
        bg: "#000000", bg2: "#0a0a10", bg3: "#000000",
        surface0: "#11111b", surface1: "#1f2229", surface2: "#2a2a36",
        overlay0: "#3a3a4a", overlay1: "#5a5f6b", overlay2: "#7f849c",
        fg: "#e8ecf1", sub: "#9aa1ab", subtext1: "#bac2de", dim: "#5a5f6b",
        border: "#1f2229",
        highlightBase: "#ffffff", shadowBase: "#000000",
        mauve: "#cba6f7", blue: "#89b4fa", sky: "#89dceb", sapphire: "#74c7ec",
        teal: "#94e2d5", green: "#a6e3a1", yellow: "#f9e2af",
        peach: "#fab387", red: "#f38ba8", pink: "#f5c2e7",
        lavender: "#b4befe", rosewater: "#f5e0dc", flamingo: "#f2cdcd", maroon: "#eba0ac",
        panelBg: "#0a0a10", panelBgAlt: "#0d0d12",
        panelTint: "#0e0e14", panelHi: "#0e0e16", panelHi2: "#0f1218",
        scrimDark: "#000000", scrimDark2: "#000000",
        accentSoft: "#7bb6f4", accentSoft2: "#5b8def", accentLink: "#9ec1ff",
        accentLight: "#cfe0ff", accentLight2: "#cfd6e0",
        accentBright: "#e7ecf3", accentText: "#e5ecf5", accentNeutral: "#e6e8ec",
        accentMuted: "#7d8896", accentMuted2: "#7a8492",
        ok: "#7ee0a8", okBright: "#3ecf5a", okSoft: "#a6e3a1", okSoft2: "#6ee7b7", okMute: "#3aa66a",
        warn: "#ffd166", warnSoft: "#fde68a", warnBright: "#facc15", warnTint: "#fce5b6",
        danger: "#ff6b6b", dangerSoft: "#ff8e8e", dangerTint: "#fca5a5", dangerStrong: "#f87171",
        violet: "#b39ddb", violetSoft: "#c084fc", violetTint: "#d8b4fe",
        white: "#ffffff", black: "#000000", gray100: "#e0e0e0", gray200: "#cccccc",
        gray300: "#aaaaaa", gray400: "#888888", gray500: "#666666", gray600: "#555555",
        gray700: "#444444", gray800: "#333333", gray900: "#252525", gray950: "#1a1a1a",
        // __P0_1B_EXTRAS__
        accentVivid: "#7c5cff", panelDeep: "#15151c", cyanBright: "#55bbff", dangerBg: "#3a1e29",
        slate400: "#94a3b8", okBg: "#3a5f50", panelDeep2: "#161a22", blueDim: "#5b9bd5",
        surface1Alt: "#3a3a4a", scrimDark3: "#0e0e14", panelDeep3: "#141821", panelDeep4: "#0f1218",
        slate500: "#6b7280", okVivid: "#4ade80", gray350: "#999999", neutral900: "#2a2a2a",
        okBg2: "#2d4a3e", dangerBg2: "#4a2d2d", dangerBg3: "#5f3a3a", blueSoft: "#a1c4fd",
        gray150: "#dddddd", slateLight: "#9aa6b2", slate300: "#9ca3af", dangerSoft2: "#ff7a7a",
        blueDeep: "#2c4a82", peachBg: "#5a3a30", peachSoft: "#ffd1a3", slate350: "#8b96a5",
        gray120: "#e6e6e6", neutral850: "#3a3a3a", fgAltLight: "#e8ecf1", cyanMid: "#5bd3ef",
        violetMid: "#a06fd0", dangerMid: "#e25c5c", amberDim: "#a89060", amberMid: "#d4a04a",
        okPale: "#9fe39f", neutral950: "#1e1e1e", neutral990: "#111111", gray110: "#eeeeee",
        gray250: "#bbbbbb", neutral955: "#1c1c1c", scrimDark4: "#0a0a1a", okBgDark: "#1e3a2a",
        warnBg: "#3a361e", warnBg2: "#3a311e", okBgDark2: "#16331f", okBright2: "#7be089",
        warnBg3: "#3a2e15", amberSoft: "#f0c764", amberStrong: "#e3a93b", dangerBgDark: "#3a1818",
        dangerSoft3: "#ff8b8b", dangerMid2: "#e25151", panelDeep5: "#101820", panelDeep6: "#1f2c3a",
        panelDeep7: "#0a1218", panelDeep8: "#1a2632", peachMid: "#d58a3e", panelDeep9: "#2a3a4a",
        blueBg: "#1e3a5f", okBgDark3: "#1a3a1a", warnBg4: "#3a2a1a", blueLink: "#003399",
        okStrong: "#22d65a", dangerHot: "#ff4466", violetBg: "#583444", gray050: "#f5f5f5",
        gray220: "#c0c0c0", gray360: "#999999", blueAction: "#1a73e8", gray450: "#777777",
        okDeep: "#2e7d32", panelMid: "#3a5068", panelMid2: "#2a3040", panelMid3: "#2a4058",
        panelMid4: "#1a2030", blueSoft2: "#64b5f6", pinkMid: "#ff6e9c", okMid: "#81c784",
        amberSoft2: "#ffd54f", cyanSoft: "#4dd0e1", amberMid2: "#ffa726", okPale2: "#c5e1a5",
        violetSoft2: "#ce93d8", slate450: "#90a4ae", dangerTint2: "#ef9a9a", tealMid: "#4db6ac",
        dangerSoft4: "#e57373", panelDeep10: "#16213e", panelDeep11: "#1e1e28", surface2Alt: "#3a3a48",
        peachVivid: "#ff8800", neutral920: "#222222", okBrightAlt: "#55dd55", dangerHotAlt: "#ee5555",
        amberHotAlt: "#ffcc66", panelTintAlt: "#222234", scrimDark5: "#101018", panelBgAlt2: "#2a2a3e",
        scrimDark6: "#0f0f17", okMid2: "#7fd17a", violetBgAlt: "#332233", neutral910: "#232323",
        neutral940: "#1f1f1f", cyanBright2: "#66ccff", scrimDark7: "#181820", slateMute: "#7d8590",
    },
};

let _variant = "mocha";
export let C = { ...PALETTES.mocha };

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
try {
    app.registerExtension({
        name: "C2C.Theme",
        settings: [
            {
                id: "c2c.theme.variant",
                name: "C2C → Theme variant",
                tooltip: "Catppuccin variant used across every C2C panel/HUD.",
                type: "combo",
                options: [
                    { value: "mocha", text: "Mocha (default dark)" },
                    { value: "latte", text: "Latte (light)" },
                    { value: "oled",  text: "OLED (true black)" },
                ],
                defaultValue: "mocha",
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
} catch (__c2cErr) { __c2cReport("_c2c_theme", __c2cErr); }
