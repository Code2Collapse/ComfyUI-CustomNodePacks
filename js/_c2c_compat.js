/**
 * _c2c_compat.js — ComfyUI menu hook compatibility (1.52+ vs legacy prototype).
 */
import { app } from "../../scripts/app.js";
import { getRuntime } from "./_c2c_runtime.js";
import { reportFailure } from "./_c2c_report.js";

const _legacyInstalled = new Set();

/** True when the front-end collects menu items via extension hooks (ComfyUI 1.52+). */
export function hasMenuHooks() {
    try {
        return typeof app.collectCanvasMenuItems === "function"
            && typeof app.collectNodeMenuItems === "function";
    } catch (_) {
        return false;
    }
}

/**
 * Legacy canvas menu patch — no-op when extension hooks exist.
 * @param {string} id unique patch id
 * @param {(opts: any[], ...args: any[]) => any[]} mergeFn receives core items; return merged array
 */
export function legacyCanvasMenu(id, mergeFn) {
    if (hasMenuHooks()) return;
    const key = `canvas:${id}`;
    if (_legacyInstalled.has(key)) return;
    const proto = (window.LGraphCanvas ?? window.LiteGraph?.LGraphCanvas)?.prototype;
    if (!proto || typeof mergeFn !== "function") return;
    getRuntime().safePatch(proto, "getCanvasMenuOptions", (orig) => function (...args) {
        const base = orig.apply(this, args);
        const opts = Array.isArray(base) ? base : [];
        try {
            const merged = mergeFn.call(this, opts, ...args);
            return Array.isArray(merged) ? merged : opts;
        } catch (e) {
            reportFailure(`legacyCanvasMenu:${id}`, e);
            return opts;
        }
    }, { id: `compat.canvas.${id}` });
    _legacyInstalled.add(key);
}

/**
 * Legacy node menu patch — no-op when extension hooks exist.
 * @param {string} id unique patch id
 * @param {(opts: any[], node: object, ...args: any[]) => any[]} mergeFn
 */
export function legacyNodeMenu(id, mergeFn) {
    if (hasMenuHooks()) return;
    const key = `node:${id}`;
    if (_legacyInstalled.has(key)) return;
    const proto = (window.LGraphCanvas ?? window.LiteGraph?.LGraphCanvas)?.prototype;
    if (!proto || typeof mergeFn !== "function") return;
    getRuntime().safePatch(proto, "getNodeMenuOptions", (orig) => function (node, ...rest) {
        const base = orig.apply(this, [node, ...rest]);
        const opts = Array.isArray(base) ? base : [];
        try {
            const merged = mergeFn.call(this, opts, node, ...rest);
            return Array.isArray(merged) ? merged : opts;
        } catch (e) {
            reportFailure(`legacyNodeMenu:${id}`, e);
            return opts;
        }
    }, { id: `compat.node.${id}` });
    _legacyInstalled.add(key);
}

/** @returns {boolean} graph safe to read (not missing / mid-configure). */
export function graphReadable() {
    try {
        const g = app?.graph;
        return !!(g && !g._configuring && Array.isArray(g._nodes));
    } catch (_) {
        return false;
    }
}

/**
 * Debounced graph-change reader: at most one serialize/read per change generation.
 * @param {() => any} readFn called when graph is readable; should not throw
 */
export function onGraphRead(readFn, { debounceMs = 250 } = {}) {
    const rt = getRuntime();
    let gen = 0;
    let lastGen = -1;
    let cached = null;
    const run = rt.guard(() => {
        if (!graphReadable()) return;
        if (lastGen === gen && cached !== null) return cached;
        lastGen = gen;
        cached = readFn();
        return cached;
    }, "onGraphRead");
    const unsub = rt.onGraphChange(() => {
        gen += 1;
        lastGen = -1;
        cached = null;
        run();
    }, { debounceMs });
    return unsub;
}
