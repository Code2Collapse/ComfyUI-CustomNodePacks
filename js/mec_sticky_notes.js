/**
 * mec_sticky_notes.js — Phase 18a: Sticky Notes with color coding
 *
 * Right-click on empty canvas → "📝 Add sticky note". Notes are stored in
 * `graph.extra.mec_sticky_notes = [{ id, x, y, w, h, color, text }]` so they
 * persist in the workflow JSON. Drawn under nodes via onDrawBackground.
 * Double-click to edit text; right-click sticky to change color or delete.
 *
 * Setting:
 *   mec.sticky_notes.enabled — bool (default true)
 */

import { app } from "../../scripts/app.js";

const COLORS = [
    { name: "Yellow", bg: "rgba(249, 226, 175, 0.20)", border: "#f9e2af" },
    { name: "Green",  bg: "rgba(166, 227, 161, 0.20)", border: "#a6e3a1" },
    { name: "Blue",   bg: "rgba(137, 180, 250, 0.20)", border: "#89b4fa" },
    { name: "Pink",   bg: "rgba(245, 194, 231, 0.20)", border: "#f5c2e7" },
    { name: "Mauve",  bg: "rgba(203, 166, 247, 0.20)", border: "#cba6f7" },
    { name: "Red",    bg: "rgba(243, 139, 168, 0.20)", border: "#f38ba8" },
];

function _settingsEnabled() {
    try { return app.ui.settings.getSettingValue("mec.sticky_notes.enabled", true); }
    catch { return true; }
}

function _store() {
    const g = app.graph;
    if (!g) return [];
    g.extra = g.extra || {};
    g.extra.mec_sticky_notes = g.extra.mec_sticky_notes || [];
    return g.extra.mec_sticky_notes;
}

function _genId() { return "s_" + Math.random().toString(36).slice(2, 10); }

function _addSticky(x, y) {
    const notes = _store();
    notes.push({
        id: _genId(),
        x: x - 80, y: y - 50,
        w: 220, h: 140,
        color: 0,
        text: "Click to edit…",
    });
    app.graph._mec_dirty = true;
    app.canvas?.setDirty?.(true, true);
}

function _hit(note, x, y) {
    return x >= note.x && y >= note.y && x <= note.x + note.w && y <= note.y + note.h;
}

function _draw(ctx) {
    if (!_settingsEnabled()) return;
    const notes = _store();
    if (!notes.length) return;
    ctx.save();
    for (const n of notes) {
        const c = COLORS[n.color % COLORS.length];
        ctx.fillStyle = c.bg;
        ctx.strokeStyle = c.border;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(n.x, n.y, n.w, n.h, 6);
        else ctx.rect(n.x, n.y, n.w, n.h);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "#cdd6f4";
        ctx.font = "13px -apple-system, Segoe UI, sans-serif";
        ctx.textBaseline = "top";
        _wrapText(ctx, n.text || "", n.x + 8, n.y + 8, n.w - 16, 16);
    }
    ctx.restore();
}

function _wrapText(ctx, text, x, y, maxWidth, lineHeight) {
    const paragraphs = String(text).split(/\r?\n/);
    let yy = y;
    for (const para of paragraphs) {
        const words = para.split(/\s+/);
        let line = "";
        for (const word of words) {
            const test = line ? line + " " + word : word;
            if (ctx.measureText(test).width > maxWidth && line) {
                ctx.fillText(line, x, yy);
                yy += lineHeight;
                line = word;
            } else {
                line = test;
            }
        }
        if (line) { ctx.fillText(line, x, yy); yy += lineHeight; }
    }
}

function _editNote(note) {
    const val = prompt("Edit sticky note:", note.text || "");
    if (val === null) return;
    note.text = val;
    app.canvas?.setDirty?.(true, true);
}

function _stickyContextMenu(note, e) {
    const opts = COLORS.map((c, i) => ({
        content: `■ ${c.name}`,
        callback: () => { note.color = i; app.canvas?.setDirty?.(true, true); },
    }));
    opts.push(null);
    opts.push({ content: "✏ Edit text", callback: () => _editNote(note) });
    opts.push({
        content: "🗑 Delete",
        callback: () => {
            const arr = _store();
            const idx = arr.indexOf(note);
            if (idx >= 0) arr.splice(idx, 1);
            app.canvas?.setDirty?.(true, true);
        },
    });
    new LiteGraph.ContextMenu(opts, { event: e });
}

function _patch() {
    if (LGraphCanvas.prototype._mecStickyPatched) return;
    LGraphCanvas.prototype._mecStickyPatched = true;

    const origBg = LGraphCanvas.prototype.onDrawBackground;
    LGraphCanvas.prototype.onDrawBackground = function (ctx) {
        if (origBg) origBg.call(this, ctx);
        try { _draw(ctx); } catch { /* ignore */ }
    };

    const origMenu = LGraphCanvas.prototype.getCanvasMenuOptions;
    LGraphCanvas.prototype.getCanvasMenuOptions = function () {
        const opts = origMenu ? origMenu.apply(this, arguments) : [];
        if (!_settingsEnabled()) return opts;
        const canvas = this;
        opts.push(null);
        opts.push({
            content: "📝 Add sticky note",
            callback: () => {
                const p = canvas.graph_mouse || [0, 0];
                _addSticky(p[0], p[1]);
            },
        });
        return opts;
    };

    // Hook mouse down to: 1) sticky context menu on right-click over sticky;
    //                     2) drag sticky;  3) double-click to edit.
    const origDown = LGraphCanvas.prototype.processMouseDown;
    LGraphCanvas.prototype.processMouseDown = function (e) {
        if (_settingsEnabled() && this.graph) {
            const [mx, my] = this.graph_mouse || [0, 0];
            // Only act when no node under cursor.
            const overNode = this.graph.getNodeOnPos
                ? this.graph.getNodeOnPos(mx, my, this.visible_nodes || this.graph._nodes)
                : null;
            if (!overNode) {
                const notes = _store();
                // Topmost (last drawn) first.
                for (let i = notes.length - 1; i >= 0; i--) {
                    const n = notes[i];
                    if (!_hit(n, mx, my)) continue;
                    if (e.button === 2) {
                        _stickyContextMenu(n, e);
                        return false;
                    }
                    if (e.detail === 2) {
                        _editNote(n);
                        return false;
                    }
                    if (e.button === 0) {
                        const off = [mx - n.x, my - n.y];
                        const move = (ev) => {
                            this.adjustMouseEvent(ev);
                            const [gx, gy] = this.graph_mouse || [0, 0];
                            n.x = gx - off[0];
                            n.y = gy - off[1];
                            this.setDirty(true, true);
                        };
                        const up = () => {
                            window.removeEventListener("mousemove", move);
                            window.removeEventListener("mouseup", up);
                        };
                        window.addEventListener("mousemove", move);
                        window.addEventListener("mouseup", up);
                        return false;
                    }
                }
            }
        }
        return origDown ? origDown.apply(this, arguments) : undefined;
    };
}

app.registerExtension({
    name: "MEC.StickyNotes",
    settings: [
        {
            id: "mec.sticky_notes.enabled",
            name: "Sticky Notes: right-click canvas → Add",
            type: "boolean",
            default: true,
        },
    ],
    async setup() {
        _patch();
        console.log("[MEC.StickyNotes] Loaded.");
    },
});
