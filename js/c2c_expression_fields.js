// c2c_expression_fields.js — Excel-style `=expr` in numeric widgets (C2C)
// ---------------------------------------------------------------------
// What it does:
//   • Any numeric (INT/FLOAT) widget will evaluate values starting with
//     `=` as a sandboxed expression.
//   • Vocabulary:
//       basic arithmetic +-*/% **,
//       Math.* whitelisted (sin/cos/log/sqrt/min/max/abs/floor/ceil/round/pow),
//       clamp(x, lo, hi), lerp(a, b, t), pi, tau, e,
//       references to other nodes: $<id>.<widget_name>
//         (e.g. =$5.width / 2) — re-evaluated each time the source changes.
//   • Sandbox: a tiny shunting-yard / recursive-descent parser. NO `eval`,
//     no `Function`, no `with`. Unknown identifiers throw clean errors.
//   • If parse fails, the field reverts to its last numeric value and a
//     toast surfaces the error.
//   • Expression is stored on the widget as `_expr` and re-evaluated when
//     ANY referenced node's value changes (250 ms debounce).
//
// License: Apache-2.0
// ---------------------------------------------------------------------

import { app } from "../../scripts/app.js";

const SETTING_ID = "c2c.expressionFields.enabled";

// ── Tiny safe expression parser ───────────────────────────────────────
const CONSTS = { pi: Math.PI, tau: Math.PI * 2, e: Math.E };
const FUNCS = {
    sin: Math.sin, cos: Math.cos, tan: Math.tan,
    asin: Math.asin, acos: Math.acos, atan: Math.atan, atan2: Math.atan2,
    log: Math.log, log2: Math.log2, log10: Math.log10, exp: Math.exp,
    sqrt: Math.sqrt, abs: Math.abs, sign: Math.sign,
    floor: Math.floor, ceil: Math.ceil, round: Math.round,
    min: Math.min, max: Math.max, pow: Math.pow,
    clamp: (x, lo, hi) => Math.min(Math.max(x, lo), hi),
    lerp: (a, b, t) => a + (b - a) * t,
    step: (edge, x) => (x < edge ? 0 : 1),
};

function tokenize(src) {
    const tk = []; let i = 0;
    while (i < src.length) {
        const c = src[i];
        if (/\s/.test(c)) { i++; continue; }
        if (/[0-9.]/.test(c)) {
            let j = i;
            while (j < src.length && /[0-9.eE+\-]/.test(src[j]) && !(src[j] === '+' || src[j] === '-' && src[j-1] !== 'e' && src[j-1] !== 'E')) j++;
            // Simpler: greedy parse
            let m = src.slice(i).match(/^[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?/);
            if (!m) throw new Error("number parse @ " + i);
            tk.push({ t: "num", v: parseFloat(m[0]) });
            i += m[0].length;
            continue;
        }
        if (/[A-Za-z_]/.test(c)) {
            let m = src.slice(i).match(/^[A-Za-z_][A-Za-z0-9_]*/);
            tk.push({ t: "id", v: m[0] });
            i += m[0].length;
            continue;
        }
        if (c === '$') {
            // $<id>.<widget>
            let m = src.slice(i).match(/^\$(-?\d+)\.([A-Za-z_][A-Za-z0-9_]*)/);
            if (!m) throw new Error("bad $ref @ " + i);
            tk.push({ t: "ref", id: m[1], w: m[2] });
            i += m[0].length;
            continue;
        }
        if ("+-*/%(),".includes(c)) { tk.push({ t: c }); i++; continue; }
        if (c === '*' && src[i+1] === '*') { tk.push({ t: "pow" }); i += 2; continue; }
        throw new Error("unexpected '" + c + "' @ " + i);
    }
    return tk;
}

function makeParser(tokens) {
    let pos = 0;
    const peek = () => tokens[pos];
    const eat = (t) => { const x = tokens[pos]; if (!x || (t && x.t !== t)) throw new Error("expected " + t); pos++; return x; };

    function parseExpr()  { return parseAddSub(); }
    function parseAddSub() {
        let l = parseMulDiv();
        while (peek() && (peek().t === "+" || peek().t === "-")) {
            const op = eat().t;
            const r = parseMulDiv();
            l = { op, l, r };
        }
        return l;
    }
    function parseMulDiv() {
        let l = parsePow();
        while (peek() && (peek().t === "*" || peek().t === "/" || peek().t === "%")) {
            const op = eat().t;
            const r = parsePow();
            l = { op, l, r };
        }
        return l;
    }
    function parsePow() {
        let l = parseUnary();
        if (peek() && peek().t === "pow") { eat("pow"); const r = parsePow(); return { op: "pow", l, r }; }
        return l;
    }
    function parseUnary() {
        if (peek() && peek().t === "-") { eat("-"); return { op: "neg", x: parseUnary() }; }
        if (peek() && peek().t === "+") { eat("+"); return parseUnary(); }
        return parsePrimary();
    }
    function parsePrimary() {
        const t = peek();
        if (!t) throw new Error("unexpected end");
        if (t.t === "num") { eat("num"); return { num: t.v }; }
        if (t.t === "ref") { eat("ref"); return { ref: { id: t.id, w: t.w } }; }
        if (t.t === "(")   { eat("("); const e = parseExpr(); eat(")"); return e; }
        if (t.t === "id")  {
            eat("id");
            if (peek() && peek().t === "(") {
                eat("(");
                const args = [];
                if (peek() && peek().t !== ")") {
                    args.push(parseExpr());
                    while (peek() && peek().t === ",") { eat(","); args.push(parseExpr()); }
                }
                eat(")");
                return { call: t.v, args };
            }
            return { sym: t.v };
        }
        throw new Error("bad token " + t.t);
    }
    const ast = parseExpr();
    if (pos < tokens.length) throw new Error("trailing tokens");
    return ast;
}

function evalAst(ast, refResolver) {
    if (ast.num != null) return ast.num;
    if (ast.sym) {
        if (ast.sym in CONSTS) return CONSTS[ast.sym];
        throw new Error("unknown identifier '" + ast.sym + "'");
    }
    if (ast.ref) return refResolver(ast.ref.id, ast.ref.w);
    if (ast.call) {
        const f = FUNCS[ast.call];
        if (!f) throw new Error("unknown function '" + ast.call + "'");
        return f(...ast.args.map((a) => evalAst(a, refResolver)));
    }
    if (ast.op === "neg") return -evalAst(ast.x, refResolver);
    const a = evalAst(ast.l, refResolver), b = evalAst(ast.r, refResolver);
    switch (ast.op) {
        case "+": return a + b;
        case "-": return a - b;
        case "*": return a * b;
        case "/": return a / b;
        case "%": return a % b;
        case "pow": return a ** b;
    }
    throw new Error("bad op");
}

function evaluate(src, refResolver) {
    const body = src.replace(/^=\s*/, "");
    const tk = tokenize(body);
    const ast = makeParser(tk);
    const v = evalAst(ast, refResolver);
    if (typeof v !== "number" || !Number.isFinite(v)) throw new Error("non-finite result");
    return v;
}

// ── Reference tracker ─────────────────────────────────────────────────
function resolveRef(id, name) {
    const node = (app.graph?._nodes || []).find((n) => String(n.id) === String(id));
    if (!node) throw new Error("no node #" + id);
    const w = (node.widgets || []).find((x) => x.name === name);
    if (!w) throw new Error("no widget '" + name + "' on #" + id);
    const v = typeof w.value === "number" ? w.value : parseFloat(w.value);
    if (Number.isNaN(v)) throw new Error("non-numeric '" + name + "' on #" + id);
    return v;
}

// ── Widget hook ───────────────────────────────────────────────────────
function hookWidget(node, w) {
    if (!w || w._c2c_expr_hooked) return;
    if (w.type !== "number" && w.type !== "INT" && w.type !== "FLOAT" && w.type !== "slider") return;
    w._c2c_expr_hooked = true;
    const origCb = w.callback;
    w.callback = function (v, ...rest) {
        if (typeof v === "string" && v.trim().startsWith("=")) {
            try {
                const out = evaluate(v, resolveRef);
                w._expr = v;
                w.value = out;
                // refresh DOM widget element if any
                if (w.inputEl) w.inputEl.value = out;
                if (typeof origCb === "function") origCb.call(this, out, ...rest);
                return;
            } catch (e) {
                try { app.extensionManager?.toast?.add({ severity: "warn", summary: "Expression error", detail: e.message, life: 2200 }); } catch { /* */ }
                return;
            }
        } else {
            // user overwrote literal — drop the expr binding.
            w._expr = undefined;
        }
        if (typeof origCb === "function") return origCb.call(this, v, ...rest);
    };
}

function rebindAll() {
    for (const node of app.graph?._nodes || []) {
        for (const w of node.widgets || []) hookWidget(node, w);
    }
}

function reevalAll() {
    for (const node of app.graph?._nodes || []) {
        for (const w of node.widgets || []) {
            if (!w._expr) continue;
            try {
                const v = evaluate(w._expr, resolveRef);
                if (v !== w.value) {
                    w.value = v;
                    if (w.inputEl) w.inputEl.value = v;
                }
            } catch { /* ignore until source becomes valid */ }
        }
    }
}

app.registerExtension({
    name: "C2C.ExpressionFields",
    async setup() {
        try {
            app.ui.settings.addSetting({
                id: SETTING_ID,
                name: "Expression fields in numeric widgets (=expr)",
                tooltip: "Type `=512*2` or `=$5.width/2` into any INT/FLOAT widget.",
                type: "boolean", defaultValue: true,
                category: ["c2c", "Editing", "Expressions"],
            });
        } catch { /* */ }
        // Initial pass + watch for node creation.
        setTimeout(rebindAll, 250);
        const orig = window.LGraph?.prototype?.add;
        if (orig && !window.LGraph.prototype._c2c_expr_patched) {
            window.LGraph.prototype.add = function (node, ...rest) {
                const r = orig.apply(this, [node, ...rest]);
                queueMicrotask(() => {
                    for (const w of node.widgets || []) hookWidget(node, w);
                });
                return r;
            };
            window.LGraph.prototype._c2c_expr_patched = true;
        }
        // Periodic re-eval (cheap; only nodes with _expr do work).
        let last = 0;
        function tick() {
            const now = performance.now();
            if (now - last > 250) { reevalAll(); last = now; }
            requestAnimationFrame(tick);
        }
        tick();
        console.log("[C2C.ExpressionFields] ready (type =expr in any numeric widget).");
    },
});
