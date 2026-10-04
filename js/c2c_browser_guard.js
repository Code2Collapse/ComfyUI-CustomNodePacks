/**
 * c2c_browser_guard.js — browser-side GPU/WebGL stability for CNP.
 *
 * - Settings slider reserves VRAM for the browser/desktop (server route).
 * - WebGL context tracking + orphan janitor (no timers; runs on new contexts).
 * - Software-renderer and canvas-farbling detection (one toast each per session).
 *
 * Idempotent: safe if ComfyUI loads this extension twice.
 */
import { app } from "../../scripts/app.js";
import { getRuntime } from "./_c2c_runtime.js";

if (globalThis.__C2C_BROWSER_GUARD__) {
    // Already installed — skip re-wrapping getContext and re-registering hooks.
} else {
    // Entries hold WEAK references: tracking a context must never keep it (or
    // its canvas) alive. A strong Map kept every orphaned context resident
    // until the janitor ran, i.e. the guard would cause the leak it exists
    // to prevent.
    const G = (globalThis.__C2C_BROWSER_GUARD__ = {
        live: new Set(),
        getContextWrapped: false,
        farbleChecked: false,
        ctxLostToasted: false,
        softwareToasted: false,
        origGetContext: null,
    });

    const ROUTE = "/c2c/memory/browser_headroom";
    const SETTING_ID = "c2c.browser.headroomGb";
    let _postTimer = 0;

    function _toastOnce(storageKey, severity, summary, detail) {
        try {
            if (sessionStorage.getItem(storageKey)) return;
            sessionStorage.setItem(storageKey, "1");
        } catch (_) { /* private mode */ }
        try {
            const toast = app.extensionManager?.toast;
            if (toast?.add) { toast.add({ severity, summary, detail, life: 8000 }); return; }
        } catch (_) { /* fall through */ }
        console.warn(`[C2C.BrowserGuard] ${summary}: ${detail}`);
    }

    function _postHeadroom(gb) {
        clearTimeout(_postTimer);
        _postTimer = setTimeout(() => {
            try {
                fetch(ROUTE, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ gb: Number(gb) || 0 }),
                }).catch(() => { /* server down / older ComfyUI */ });
            } catch (_) { /* never break the page */ }
        }, 300);
    }

    const _deref = (r) => (r && typeof r.deref === "function" ? r.deref() : r);

    /** Drop entries whose context was collected or lost. */
    function _pruneLost() {
        for (const entry of G.live) {
            const ctx = _deref(entry.ctxRef);
            try {
                if (!ctx || (ctx.isContextLost && ctx.isContextLost())) G.live.delete(entry);
            } catch (_) {
                G.live.delete(entry);
            }
        }
    }

    function _onContextLost() {
        if (!G.ctxLostToasted) {
            G.ctxLostToasted = true;
            _toastOnce(
                "c2c.browser.ctxlost",
                "warn",
                "A 3D view lost its GPU memory",
                "If this keeps happening, raise C2C → Keep VRAM free for the browser (Settings).",
            );
        }
    }

    function _checkSoftwareRenderer(ctx) {
        if (G.softwareToasted) return;
        try {
            const dbg = ctx.getExtension("WEBGL_debug_renderer_info");
            const renderer = dbg
                ? String(ctx.getParameter(dbg.UNMASKED_RENDERER_WEBGL) || "")
                : "";
            if (/swiftshader|llvmpipe|software|basic render/i.test(renderer)) {
                G.softwareToasted = true;
                _toastOnce(
                    "c2c.browser.software",
                    "warn",
                    "WebGL is running without the GPU (software)",
                    "Turn on hardware acceleration in the browser settings; 3D views will be slow.",
                );
            }
        } catch (_) { /* extension blocked */ }
    }

    function runJanitor() {
        _pruneLost();
        for (const entry of G.live) {
            const canvas = _deref(entry.canvasRef);
            try {
                if (canvas && canvas.isConnected) {
                    entry.disconnectedPasses = 0;
                } else {
                    entry.disconnectedPasses = (entry.disconnectedPasses || 0) + 1;
                }
            } catch (_) {
                entry.disconnectedPasses = (entry.disconnectedPasses || 0) + 1;
            }
        }
        if (G.live.size < 12) return;
        for (const entry of G.live) {
            const ctx = _deref(entry.ctxRef), canvas = _deref(entry.canvasRef);
            try {
                if (!ctx) { G.live.delete(entry); continue; }
                if (canvas && canvas.isConnected) continue;
                if ((entry.disconnectedPasses || 0) < 2) continue;
                const ext = ctx.getExtension("WEBGL_lose_context");
                if (ext && typeof ext.loseContext === "function") {
                    ext.loseContext();
                    console.debug("[C2C.BrowserGuard] released orphaned WebGL context");
                }
                G.live.delete(entry);
            } catch (_) { /* keep going */ }
        }
    }

    function trackWebGLContext(canvas, ctx) {
        if (!ctx || !canvas) return;
        _pruneLost();
        let known = false;
        for (const e of G.live) if (_deref(e.ctxRef) === ctx) { known = true; break; }
        if (!known) {
            const weak = (o) => (typeof WeakRef === "function" ? new WeakRef(o) : o);
            G.live.add({ ctxRef: weak(ctx), canvasRef: weak(canvas), disconnectedPasses: 0 });
            try {
                canvas.addEventListener("webglcontextlost", (e) => {
                    try { e.preventDefault(); } catch (_) { /* */ }
                    _onContextLost();
                });
            } catch (_) { /* */ }
            _checkSoftwareRenderer(ctx);
        }
        runJanitor();
    }

    function wrapGetContext() {
        if (G.getContextWrapped) return;
        G.origGetContext = HTMLCanvasElement.prototype.getContext;
        getRuntime().safePatch(HTMLCanvasElement.prototype, "getContext", (orig) => function (type, attrs) {
            const ctx = orig.call(this, type, attrs);
            if (ctx && typeof type === "string" && /webgl/i.test(type)) {
                trackWebGLContext(this, ctx);
            }
            return ctx;
        }, { id: "browserguard.getContext" });
        G.getContextWrapped = true;
    }

    function checkCanvasFarbling() {
        const W = 64;
        const H = 64;
        const canvas = document.createElement("canvas");
        canvas.width = W;
        canvas.height = H;
        const ctx = canvas.getContext("2d");
        if (!ctx) return false;
        const expected = new Uint8ClampedArray(W * H * 4);
        for (let y = 0; y < H; y++) {
            for (let x = 0; x < W; x++) {
                const i = (y * W + x) * 4;
                expected[i] = (x * 4) & 255;
                expected[i + 1] = (y * 4) & 255;
                expected[i + 2] = 128;
                expected[i + 3] = 255;
                ctx.fillStyle = `rgba(${expected[i]},${expected[i + 1]},${expected[i + 2]},1)`;
                ctx.fillRect(x, y, 1, 1);
            }
        }
        const data = ctx.getImageData(0, 0, W, H).data;
        for (let i = 0; i < expected.length; i++) {
            if (data[i] !== expected[i]) return true;
        }
        return false;
    }

    function runFarblingCheck() {
        if (G.farbleChecked) return;
        G.farbleChecked = true;
        try {
            if (checkCanvasFarbling()) {
                globalThis.__C2C_CANVAS_FARBLED = true;
                _toastOnce(
                    "c2c.browser.farble",
                    "warn",
                    "This browser changes canvas pixels when they are read",
                    "Brave Shields / fingerprinting protection can add faint noise to masks and paint strokes saved from C2C editors. Turn Shields off for this ComfyUI address.",
                );
            }
        } catch (_) { /* never break setup */ }
    }

    wrapGetContext();

    app.registerExtension({
        name: "C2C.BrowserGuard",
        settings: [
            {
                id: SETTING_ID,
                name: "Keep VRAM free for the browser (GB)",
                category: ["c2c", "Performance", "Browser stability"],
                type: "slider",
                attrs: { min: 0, max: 4, step: 0.25 },
                defaultValue: 0,
                tooltip:
                    "Reserves GPU memory for the browser and desktop so 3D views are less likely to lose WebGL context. 0 = ComfyUI's own default.",
                onChange: (v) => { _postHeadroom(v); },
            },
        ],
        async setup() {
            runFarblingCheck();
        },
    });

    G.runJanitor = runJanitor;
    G.trackWebGLContext = trackWebGLContext;
    G.wrapGetContext = wrapGetContext;
    G.checkCanvasFarbling = checkCanvasFarbling;
    G._toastOnce = _toastOnce;
}

export const __test = globalThis.__C2C_BROWSER_GUARD__ || {};
