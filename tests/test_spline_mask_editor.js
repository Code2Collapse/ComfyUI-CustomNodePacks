/**
 * Headless test harness for js/spline_mask_editor.js.
 *
 * Runs in plain Node.js — no browser, no ComfyUI, no canvas package.
 * We mock just enough of the LiteGraph / Canvas2D surface that the
 * editor module depends on, then drive its public methods and assert
 * the resulting state.
 *
 * Run:  node tests/test_spline_mask_editor.js
 */
"use strict";
const path = require("path");
const fs = require("fs");
const Module = require("module");
const assert = require("assert");

// ── 1. Stub `../../scripts/app.js` (registerExtension capture) ──────
const captured = { ext: null };
const appStub = {
    app: {
        registerExtension(ext) { captured.ext = ext; },
    },
};

// Hijack require() so the editor's `import { app } from "../../scripts/app.js"`
// (after our shim → require) resolves to our stub.
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (req, parent, ...rest) {
    if (req.endsWith("scripts/app.js") || req === "../../scripts/app.js") {
        return path.join(__dirname, "__app_stub__.js");
    }
    return origResolve.call(this, req, parent, ...rest);
};
require.cache[path.join(__dirname, "__app_stub__.js")] = {
    id: path.join(__dirname, "__app_stub__.js"),
    filename: path.join(__dirname, "__app_stub__.js"),
    loaded: true,
    exports: appStub,
};

// ── 2. Read the editor source and shim ESM → CommonJS ──────────────
const SRC = path.join(__dirname, "..", "js", "spline_mask_editor.js");
let src = fs.readFileSync(SRC, "utf8");
// Convert the lone ESM import line into a CJS require of our stub.
src = src.replace(
    /import\s*\{\s*app\s*\}\s*from\s*["'][^"']+app\.js["'];?/,
    'const { app } = require("../../scripts/app.js");',
);

// Eval inside an isolated CJS-style module wrapper.
const tmpFile = path.join(__dirname, "__spline_editor_cjs__.js");
fs.writeFileSync(tmpFile, src);
try {
    require(tmpFile);
} finally {
    fs.unlinkSync(tmpFile);
}

assert.ok(captured.ext, "registerExtension was never called");
assert.strictEqual(captured.ext.name, "Comfy.MEC.SplineMaskEditor");

// ── 3. Mock Canvas2D context (records every draw call) ─────────────
function makeCtx() {
    const calls = [];
    const rec = (name) => (...args) => calls.push([name, ...args]);
    return new Proxy({ calls }, {
        get(target, prop) {
            if (prop === "calls") return calls;
            // properties Canvas code reads back
            if (["fillStyle", "strokeStyle", "lineWidth", "font",
                 "textAlign", "textBaseline"].includes(prop)) {
                return target[prop] ?? "";
            }
            // mutation of the same properties
            return target[prop] ?? rec(prop);
        },
        set(target, prop, value) { target[prop] = value; return true; },
    });
}

// ── 4. Mock LiteGraph node ─────────────────────────────────────────
function makeNode() {
    const widgets = [
        { name: "spline_data", value: "[]" },
        { name: "mask_color", value: "#ffffff" },
        { name: "mask_opacity", value: 1 },
    ];
    return {
        comfyClass: "SplineMaskEditorMEC",
        pos: [800, 200],
        size: [400, 600],
        widgets,
        properties: {},
        addCustomWidget(w) { widgets.push(w); return w; },
        setDirtyCanvas() {},
    };
}

// ── 5. Run the tests ────────────────────────────────────────────────
let pass = 0, fail = 0;
function test(name, fn) {
    try { fn(); console.log("  ✓", name); pass++; }
    catch (e) { console.error("  ✗", name, "\n     ", e.message); fail++; }
}

console.log("spline_mask_editor.js");

test("nodeCreated registers a custom widget", () => {
    const node = makeNode();
    captured.ext.nodeCreated(node);
    const w = node.widgets.find(x => x.name === "spline_editor_canvas");
    assert.ok(w, "spline_editor_canvas widget must be added");
    assert.strictEqual(typeof w.draw, "function");
    assert.strictEqual(typeof w.onMouseDown, "function");
});

test("draw uses widget-LOCAL coords (NOT absolute node.pos)", () => {
    // This is the regression test for the bug that left the canvas
    // dead in the UI: passing node.pos[0] as widgetX caused all
    // fill/clip rects to land outside the node body.
    const node = makeNode();
    captured.ext.nodeCreated(node);
    const w = node.widgets.find(x => x.name === "spline_editor_canvas");
    const ctx = makeCtx();
    w.draw(ctx, node, 400, 30, 440);

    // Find every fillRect/rect call and assert X is 0 (widget-local),
    // not node.pos[0] (= 800).
    for (const call of ctx.calls) {
        if (call[0] === "fillRect" || call[0] === "rect") {
            const x = call[1];
            assert.ok(
                x < node.pos[0],
                `fillRect/rect drew at absolute X=${x} (>= node.pos[0]=${node.pos[0]}); ` +
                "this is the canvas-outside-node bug.",
            );
        }
    }
});

test("addPoint + serialize writes pixel coords to spline_data widget", () => {
    const node = makeNode();
    captured.ext.nodeCreated(node);
    // Force a draw so previewBounds / canvas dims initialise.
    const ctx = makeCtx();
    const w = node.widgets.find(x => x.name === "spline_editor_canvas");
    w.draw(ctx, node, 400, 30, 440);

    // Simulate three left-clicks inside the canvas area.
    const events = { button: 0, shiftKey: false, ctrlKey: false, altKey: false };
    // Click in widget-local coords; pos[1] must be > TOOLBAR_H (32).
    w.onMouseDown(events, [120, 200], node);
    w.onMouseDown(events, [220, 200], node);
    w.onMouseDown(events, [180, 320], node);

    const splineData = node.widgets.find(x => x.name === "spline_data").value;
    const parsed = JSON.parse(splineData);
    assert.ok(Array.isArray(parsed) && parsed.length >= 1, "spline_data must serialise");
    const pts = parsed[0].points;
    assert.strictEqual(pts.length, 3, `expected 3 control points, got ${pts.length}`);
    for (const p of pts) {
        assert.strictEqual(typeof p.x, "number");
        assert.strictEqual(typeof p.y, "number");
    }
});

test("properties persist normalised (0..1) coords for save/load", () => {
    const node = makeNode();
    captured.ext.nodeCreated(node);
    const ctx = makeCtx();
    const w = node.widgets.find(x => x.name === "spline_editor_canvas");
    w.draw(ctx, node, 400, 30, 440);
    w.onMouseDown({ button: 0 }, [120, 200], node);
    w.onMouseDown({ button: 0 }, [220, 200], node);
    w.onMouseDown({ button: 0 }, [180, 320], node);

    const flat = node.properties.spline_points;
    assert.ok(Array.isArray(flat), "properties.spline_points must be an array");
    const real = flat.filter(p => p !== null);
    assert.ok(real.length >= 3);
    for (const [nx, ny] of real) {
        assert.ok(nx >= 0 && nx <= 1, `nx out of [0,1]: ${nx}`);
        assert.ok(ny >= 0 && ny <= 1, `ny out of [0,1]: ${ny}`);
    }
});

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail > 0 ? 1 : 0);
