/**
 * mec_maskops_health.js — at-a-glance backend health pills on MaskOpsMEC.
 *
 * The MaskOpsMEC node lets users pick a `segmenter` and a `matter`, but the
 * readiness of each backend is only visible after opening the dropdown
 * (entries are tagged with `  [missing-deps]` / `  [experimental]`). This
 * extension surfaces that information on the node itself: two small status
 * pills sit at the right end of the title bar (one for the segmenter, one
 * for the matter), colored green for "ready", amber for "experimental",
 * and red for "missing-deps". Hovering shows a tooltip; clicking the pill
 * copies a hint string to the clipboard. They never cover the title: on a
 * narrow node they shorten to "S ✓" / "M ✓", and narrower still they hide.
 *
 * Implementation notes
 * --------------------
 * - No new HTTP route is needed: the STATUS suffix is already embedded in
 *   the dropdown options assembled by ``_segmenter_choices`` / ``_matter_choices``
 *   in ``mask_matting/node.py``. We just parse it back out on the JS side.
 * - Drawing is done in ``onDrawForeground`` so pills follow the node when
 *   it's dragged, resized, or collapsed. When the node is collapsed we draw
 *   a single compact dot to the right of the title.
 * - Hit-testing uses ``onMouseDown``; clicks outside the pills fall through
 *   to LiteGraph as normal.
 *
 * License: Apache-2.0
 */

import { app } from "../../scripts/app.js";
import { C, green } from "./_c2c_theme.js";

const TARGET = "MaskOpsMEC";

// ── status → visual ────────────────────────────────────────────────────────
const STATUS_STYLE = {
    "ready":         { bg: "var(--c2c-green)", fg: "var(--c2c-bg3)", label: "ready"   },
    "experimental":  { bg: "var(--c2c-yellow)", fg: "var(--c2c-bg3)", label: "experim." },
    "missing-deps":  { bg: "var(--c2c-red)", fg: "var(--c2c-bg3)", label: "missing-deps" },
    "not-ready":     { bg: "var(--c2c-red)", fg: "var(--c2c-bg3)", label: "not-ready"   },
};
const READY_DOT = C.green;

const HINT_FOR = {
    "ready":         (k) => `${k} backend is installed and ready.`,
    "experimental":  (k) => `${k} backend is marked experimental — may be unstable.`,
    "missing-deps":  (k) => `${k} backend is missing its Python dependency. Install it via the C2C Dep-Check sidebar.`,
    "not-ready":     (k) => `${k} backend reported not-ready. Check the ComfyUI console for details.`,
};

// ── helpers ────────────────────────────────────────────────────────────────

/** Pull `name` and `[badge]` out of a choice string like `sam2.1  [missing-deps]`. */
function _parseChoice(value) {
    if (typeof value !== "string") return { name: "", status: "ready" };
    const m = value.match(/^(.*?)(?:\s{2,}\[([a-z\-]+)\])?\s*$/i);
    if (!m) return { name: value.trim(), status: "ready" };
    const name = (m[1] || "").trim();
    const status = (m[2] || "ready").trim().toLowerCase();
    return { name, status };
}

function _widgetValue(node, widgetName) {
    const w = (node.widgets || []).find((w) => w.name === widgetName);
    return w ? w.value : null;
}

function _pillColors(status) {
    return STATUS_STYLE[status] || STATUS_STYLE["ready"];
}

const PILL_FONT = "10px ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif";
const PILL_PAD_X = 6, PILL_H = 16, PILL_GAP = 6;

function _pillText(kind, name, status, compact) {
    const isReady = status === "ready";
    const labelKind = kind === "seg" ? "S" : "M";
    if (compact) return isReady ? `${labelKind} ✓` : `${labelKind} •`;
    return isReady ? `${labelKind}:${name} ✓` : `${labelKind}:${name} • ${_pillColors(status).label}`;
}

function _pillWidth(ctx, text) {
    ctx.save();
    ctx.font = PILL_FONT;
    const w = Math.ceil(ctx.measureText(text).width) + PILL_PAD_X * 2;
    ctx.restore();
    return w;
}

/** Draw a single pill at (x,y). Returns its right edge. */
function _drawPill(ctx, x, y, text, status) {
    const style = _pillColors(status);
    const isReady = status === "ready";

    ctx.save();
    ctx.font = PILL_FONT;
    const padX = PILL_PAD_X, h = PILL_H;
    const w = Math.ceil(ctx.measureText(text).width) + padX * 2;

    // Background pill
    ctx.fillStyle = isReady ? "rgba(166,227,161,0.18)" : style.bg;
    ctx.beginPath();
    const r = 8;
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y,     x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x,     y + h, r);
    ctx.arcTo(x,     y + h, x,     y,     r);
    ctx.arcTo(x,     y,     x + w, y,     r);
    ctx.closePath();
    ctx.fill();

    if (isReady) {
        // Subtle outline for ready state
        ctx.strokeStyle = "rgba(166,227,161,0.55)";
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.fillStyle = READY_DOT;
    } else {
        ctx.fillStyle = style.fg;
    }
    ctx.textBaseline = "middle";
    ctx.fillText(text, x + padX, y + h / 2 + 0.5);
    ctx.restore();

    return { right: x + w, width: w, height: h };
}

/** Compute the pill geometry for both widgets. Cached on the node as
 *  `node.__mec_health_geom` so onMouseDown can hit-test without recomputing. */
function _drawHealthPills(node, ctx) {
    if (node.flags?.collapsed) return;
    const seg = _parseChoice(_widgetValue(node, "segmenter"));
    const mat = _parseChoice(_widgetValue(node, "matter"));
    if (!seg.name && !mat.name) return;

    // Right end of the title bar, vertically centred in it. (They used to sit
    // at y=-18 from the left edge - inside the title bar, on top of the title
    // text.) Title-bar space is y in [-titleH, 0].
    const titleH = globalThis.LiteGraph?.NODE_TITLE_HEIGHT || 30;
    const y = -Math.round((titleH + PILL_H) / 2);
    const items = [];
    if (seg.name) items.push({ kind: "seg", geomKind: "segmenter", name: seg.name, status: seg.status });
    if (mat.name && mat.name !== "none") items.push({ kind: "mat", geomKind: "matter", name: mat.name, status: mat.status });

    ctx.save();
    ctx.font = `${globalThis.LiteGraph?.NODE_TEXT_SIZE || 14}px ${globalThis.LiteGraph?.NODE_FONT || "sans-serif"}`;
    const titleRight = titleH + ctx.measureText(node.getTitle?.() || node.title || "").width + 10;
    ctx.restore();
    const room = node.size[0] - 8 - titleRight;

    let texts = null;
    for (const compact of [false, true]) {
        const t = items.map((it) => _pillText(it.kind, it.name, it.status, compact));
        const total = t.reduce((a, s) => a + _pillWidth(ctx, s), 0) + PILL_GAP * (t.length - 1);
        if (total <= room) { texts = t; break; }
    }
    const geom = [];
    if (texts) {
        const total = texts.reduce((a, s) => a + _pillWidth(ctx, s), 0) + PILL_GAP * (texts.length - 1);
        let cursor = node.size[0] - 8 - total;
        items.forEach((it, i) => {
            const g = _drawPill(ctx, cursor, y, texts[i], it.status);
            geom.push({ kind: it.geomKind, name: it.name, status: it.status,
                        x: cursor, y, w: g.width, h: g.height });
            cursor = g.right + PILL_GAP;
        });
    }
    node.__mec_health_geom = geom;
}

/** Hit-test the cached pill geometry and surface a tooltip / clipboard action. */
function _onPillClick(node, localPos) {
    const geom = node.__mec_health_geom;
    if (!Array.isArray(geom) || !geom.length) return false;
    const [lx, ly] = localPos;
    for (const g of geom) {
        if (lx >= g.x && lx <= g.x + g.w && ly >= g.y && ly <= g.y + g.h) {
            const hint = (HINT_FOR[g.status] || HINT_FOR["ready"])(g.name);
            try {
                navigator.clipboard?.writeText(hint);
            } catch { /* clipboard may not be allowed; ignore */ }
            // ComfyUI's toast system (best-effort — versions vary):
            try {
                app.extensionManager?.toast?.add({
                    severity: g.status === "ready" ? "success"
                            : g.status === "missing-deps" ? "error" : "warn",
                    summary: `${node.title || node.type}`,
                    detail: hint,
                    life: 5000,
                });
            } catch { /* not all builds expose this; fall back silently */ }
            return true;
        }
    }
    return false;
}

// ── extension ──────────────────────────────────────────────────────────────
app.registerExtension({
    name: "C2C.MaskOps.HealthBadge",
    async beforeRegisterNodeDef(nodeType, nodeData, _appRef) {
        if (nodeData?.name !== TARGET) return;

        const origDraw = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            const r = origDraw?.apply(this, arguments);
            try { _drawHealthPills(this, ctx); } catch (e) {
                // Never let badge drawing crash the canvas.
                console.warn("[MEC.MaskOps.HealthBadge] draw failed:", e);
            }
            return r;
        };

        const origMouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function (e, localPos, graphCanvas) {
            try {
                if (_onPillClick(this, localPos)) {
                    // Mark dirty so the next frame re-renders pills (no visual change,
                    // but keeps LiteGraph's invalidation in sync).
                    this.setDirtyCanvas?.(true, true);
                    return true;
                }
            } catch (e2) {
                console.warn("[MEC.MaskOps.HealthBadge] click failed:", e2);
            }
            return origMouseDown?.apply(this, arguments);
        };
    },
});
