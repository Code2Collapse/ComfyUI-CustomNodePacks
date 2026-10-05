/** Resolved theme colours for Canvas2D (never var() in fillStyle). */

const _FALLBACK = {
    bg: "#1e1e2e",
    panel: "#313244",
    border: "#45475a",
    text: "#cdd6f4",
    sub: "#9399b2",
    accent: "#a6e3a1",
    accent2: "#89b4fa",
    warn: "#fab387",
    danger: "#f38ba8",
    white: "#ffffff",
    black: "#000000",
};

const _TOKEN = {
    bg: "--c2c-bg",
    panel: "--c2c-surface0",
    border: "--c2c-surface2",
    text: "--c2c-fg",
    sub: "--c2c-sub",
    accent: "--c2c-green",
    accent2: "--c2c-blue",
    warn: "--c2c-yellow",
    danger: "--c2c-red",
    white: "--c2c-white",
    black: "--c2c-black",
};

export const C = new Proxy(_FALLBACK, {
    get(target, key) {
        const tok = _TOKEN[key];
        if (tok) {
            try {
                const v = getComputedStyle(document.documentElement).getPropertyValue(tok).trim();
                if (v) return v;
            } catch (_) { /* ignore */ }
        }
        return target[key];
    },
});

export function hexToRgb(hex) {
    const h = String(hex).replace("#", "");
    if (h.length < 6) return { r: 255, g: 255, b: 255 };
    return {
        r: parseInt(h.slice(0, 2), 16),
        g: parseInt(h.slice(2, 4), 16),
        b: parseInt(h.slice(4, 6), 16),
    };
}
