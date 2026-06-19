/**
 * c2c_autocheckpoint.js — IndexedDB ring buffer of recent workflow states.
 *
 * Snapshots are stored:
 *   - On every Queue Prompt (before execution).
 *   - On graph "node added"/"node removed" events, throttled to 1 every 30 s.
 *   - Manually via the "Snapshot now" button in the picker.
 *
 * Snapshots include:
 *   {
 *     id:    "ck_<ms>_<rand>",
 *     ts:    Date.now(),
 *     label: "auto: before queue" | "auto: graph edit" | "manual",
 *     graph: <result of app.graph.serialize()>,
 *     thumb: <PNG dataURL 256x144 of canvas at snapshot time>,
 *     node_count, link_count, size_bytes
 *   }
 *
 * Ring is bounded to 20 entries (configurable). Oldest are pruned FIFO.
 *
 * Picker UI: top-left floating button "↺ History" (can be hidden via setting),
 * also accessible via command "C2C: Open auto-checkpoint history".
 */
import { app } from "../../scripts/app.js";
import { c2cConfirm, c2cAlert } from "./_c2c_dialog.js";
const DB_NAME = "c2c_autocheckpoint";
const STORE = "snapshots";
const DB_VERSION = 1;
const RING_MAX_DEFAULT = 20;
const SETTING_RING = "c2c.autocheckpoint.ringSize";
const SETTING_BTN = "c2c.autocheckpoint.showButton";
const SETTING_ENABLED = "c2c.autocheckpoint.enabled";
const THROTTLE_MS = 30_000;

// Theme-aware: surfaces below use CSS custom properties emitted by _c2c_theme.js
// (--c2c-bg/bg2/fg/sub/border/mauve/red/green/shadowBase). Canvas thumbnail paint
// resolves --c2c-bg2 at draw time via getComputedStyle so snapshots taken under
// any variant render their backdrop in the active palette.
function _cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// ============================================================ IndexedDB layer
function openDB() {
    return new Promise((res, rej) => {
        const req = indexedDB.open(DB_NAME, DB_VERSION);
        req.onupgradeneeded = () => {
            const db = req.result;
            if (!db.objectStoreNames.contains(STORE)) {
                const os = db.createObjectStore(STORE, { keyPath: "id" });
                os.createIndex("ts", "ts", { unique: false });
            }
        };
        req.onsuccess = () => res(req.result);
        req.onerror = () => rej(req.error);
    });
}

async function dbAll() {
    const db = await openDB();
    return new Promise((res, rej) => {
        const tx = db.transaction(STORE, "readonly");
        const req = tx.objectStore(STORE).getAll();
        req.onsuccess = () => res(req.result || []);
        req.onerror = () => rej(req.error);
    });
}

async function dbPut(snap) {
    const db = await openDB();
    return new Promise((res, rej) => {
        const tx = db.transaction(STORE, "readwrite");
        tx.objectStore(STORE).put(snap);
        tx.oncomplete = () => res();
        tx.onerror = () => rej(tx.error);
    });
}

async function dbDelete(id) {
    const db = await openDB();
    return new Promise((res, rej) => {
        const tx = db.transaction(STORE, "readwrite");
        tx.objectStore(STORE).delete(id);
        tx.oncomplete = () => res();
        tx.onerror = () => rej(tx.error);
    });
}

async function dbClear() {
    const db = await openDB();
    return new Promise((res, rej) => {
        const tx = db.transaction(STORE, "readwrite");
        tx.objectStore(STORE).clear();
        tx.oncomplete = () => res();
        tx.onerror = () => rej(tx.error);
    });
}

async function pruneRing(limit) {
    const all = await dbAll();
    all.sort((a, b) => a.ts - b.ts);
    while (all.length > limit) {
        const old = all.shift();
        await dbDelete(old.id);
    }
}

// =================================================================== snapshot
let _lastSnapAt = 0;
let _isSnapping = false; // Concurrency lock to prevent multiple heavy toDataURL calls

function _makeThumbnail() {
    // Grab the litegraph canvas if present, downscale to 256x144.
    const canvas = app.canvas?.canvas;
    if (!canvas) return "";
    try {
        const tmp = document.createElement("canvas");
        tmp.width = 256; tmp.height = 144;
        const ctx = tmp.getContext("2d");
        const bg2 = _cssVar("--c2c-bg2");
        if (bg2) { ctx.fillStyle = bg2; ctx.fillRect(0, 0, 256, 144); }
        ctx.drawImage(canvas, 0, 0, 256, 144);
        // JPEG (entropy-coded) encodes ~5-10x faster than PNG (lossless deflate).
        // The profiler showed PNG toDataURL at ~28% main-thread self-time; the
        // thumbnail is only ever shown scaled-down in the picker, so lossy is fine.
        return tmp.toDataURL("image/jpeg", 0.6);
    } catch (_) { return ""; }
}

export async function snapshot(label = "auto") {
    const enabled = app.ui?.settings?.getSettingValue(SETTING_ENABLED, true);
    if (!enabled) return null;
    
    // FIX: If a snapshot is already running, immediately bail out.
    // This prevents pasting 20 nodes from triggering 20 concurrent toDataURL freezes.
    if (_isSnapping) return null;
    _isSnapping = true;
    
    try {
        const graph = app.graph?.serialize?.();
        if (!graph) return null;
        const json = JSON.stringify(graph);
        const snap = {
            id: `ck_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
            ts: Date.now(),
            label,
            graph,
            thumb: _makeThumbnail(),
            node_count: (graph.nodes || []).length,
            link_count: (graph.links || []).length,
            size_bytes: json.length,
        };
        await dbPut(snap);
        const ringSize = app.ui?.settings?.getSettingValue(SETTING_RING, RING_MAX_DEFAULT) || RING_MAX_DEFAULT;
        await pruneRing(ringSize);
        _lastSnapAt = snap.ts;
        return snap;
    } catch (exc) {
        console.warn("[c2c.autocheckpoint] snapshot failed:", exc);
        return null;
    } finally {
        // Release the lock so the next snapshot can run
        _isSnapping = false;
    }
}

async function restore(id) {
    const all = await dbAll();
    const snap = all.find(x => x.id === id);
    if (!snap) return false;
    try {
        // Snapshot the CURRENT graph first so restore is reversible
        await snapshot("auto: pre-restore");
        await app.loadGraphData(snap.graph);
        return true;
    } catch (exc) {
        c2cAlert("Restore failed: " + exc.message);
        return false;
    }
}

// =================================================================== picker UI
const PICKER_ID = "c2c-autockpt-picker";
const BTN_ID = "c2c-autockpt-btn";

async function openPicker() {
    closePicker();
    const all = await dbAll();
    all.sort((a, b) => b.ts - a.ts);

    const back = document.createElement("div");
    back.id = PICKER_ID;
    back.style.cssText =
        `position:fixed;inset:0;z-index:var(--c2c-z-modal);background:color-mix(in srgb, var(--c2c-shadowBase) 55%, transparent);
         display:flex;align-items:center;justify-content:center;`;
    const card = document.createElement("div");
    card.style.cssText =
        `background:var(--c2c-bg);color:var(--c2c-fg);border:1px solid var(--c2c-border);
         border-radius:8px;padding:14px;width:720px;max-height:80vh;overflow:auto;
         font-family:ui-sans-serif,system-ui,sans-serif;font-size:12px;
         box-shadow:0 12px 40px color-mix(in srgb, var(--c2c-shadowBase) 60%, transparent);`;
    card.innerHTML =
        `<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
           <h3 style="margin:0;color:var(--c2c-mauve);font-size:14px;letter-spacing:0.5px;text-transform:uppercase">Auto-checkpoints</h3>
           <span style="color:var(--c2c-sub);font-size:11px">last ${all.length} workflow snapshots</span>
           <span style="flex:1"></span>
           <button id="snap-now" style="background:var(--c2c-mauve);color:var(--c2c-bg);border:none;border-radius:4px;padding:4px 10px;cursor:pointer;font-size:11px">Snapshot now</button>
           <button id="clear-all" style="background:var(--c2c-bg2);color:var(--c2c-red);border:1px solid var(--c2c-red);border-radius:4px;padding:4px 10px;cursor:pointer;font-size:11px">Clear all</button>
           <button id="close-x" style="background:transparent;color:var(--c2c-sub);border:none;font-size:18px;cursor:pointer">×</button>
         </div>`;
    if (all.length === 0) {
        card.insertAdjacentHTML("beforeend", `<p style="color:var(--c2c-sub)">No checkpoints yet. Queue a prompt or click "Snapshot now".</p>`);
    } else {
        const grid = document.createElement("div");
        grid.style.cssText = "display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px;";
        all.forEach(snap => {
            const card2 = document.createElement("div");
            card2.style.cssText =
                `background:var(--c2c-bg2);border:1px solid var(--c2c-border);border-radius:6px;
                 padding:6px;cursor:pointer;transition:border-color 0.15s;`;
            card2.onmouseenter = () => card2.style.borderColor = "var(--c2c-mauve)";
            card2.onmouseleave = () => card2.style.borderColor = "var(--c2c-border)";
            const dt = new Date(snap.ts);
            card2.innerHTML =
                `<img src="${snap.thumb || ""}" style="width:100%;height:120px;object-fit:cover;background:var(--c2c-bg);border-radius:3px"/>
                 <div style="margin-top:4px;font-size:11px">${snap.label}</div>
                 <div style="color:var(--c2c-sub);font-size:10px">${dt.toLocaleString()}</div>
                 <div style="color:var(--c2c-sub);font-size:10px">${snap.node_count} nodes · ${(snap.size_bytes/1024).toFixed(1)} KB</div>
                 <div style="margin-top:4px;display:flex;gap:4px">
                   <button class="ckpt-restore" data-id="${snap.id}" style="flex:1;background:var(--c2c-mauve);color:var(--c2c-bg);border:none;border-radius:3px;padding:3px;cursor:pointer;font-size:10px">Restore</button>
                   <button class="ckpt-del" data-id="${snap.id}" style="background:var(--c2c-bg);color:var(--c2c-red);border:1px solid var(--c2c-red);border-radius:3px;padding:3px 6px;cursor:pointer;font-size:10px">×</button>
                 </div>`;
            grid.appendChild(card2);
        });
        card.appendChild(grid);
    }
    back.appendChild(card);
    document.body.appendChild(back);

    back.querySelector("#close-x").onclick = closePicker;
    back.querySelector("#snap-now").onclick = async () => { await snapshot("manual"); closePicker(); openPicker(); };
    back.querySelector("#clear-all").onclick = async () => {
        if (await c2cConfirm("Delete all checkpoints?")) { await dbClear(); closePicker(); openPicker(); }
    };
    back.addEventListener("click", e => { if (e.target === back) closePicker(); });
    card.querySelectorAll(".ckpt-restore").forEach(b => b.onclick = async () => {
        if (await c2cConfirm("Restore this checkpoint? Your current graph will be snapshotted first.")) {
            const ok = await restore(b.dataset.id);
            if (ok) closePicker();
        }
    });
    card.querySelectorAll(".ckpt-del").forEach(b => b.onclick = async () => {
        await dbDelete(b.dataset.id); closePicker(); openPicker();
    });
}

function closePicker() {
    document.getElementById(PICKER_ID)?.remove();
}

function ensureButton() {
    const show = app.ui?.settings?.getSettingValue(SETTING_BTN, true);
    let btn = document.getElementById(BTN_ID);
    if (!show) { btn?.remove(); return; }
    if (btn) return;
    btn = document.createElement("button");
    btn.id = BTN_ID;
    btn.title = "C2C — open auto-checkpoint history";
    btn.textContent = "↺ History";
    // Positioning is delegated to __c2cTopDock so this button reflows below
    // ComfyUI's top chrome and never collides with the workflow-tabs row.
    btn.style.cssText =
        `background:var(--c2c-bg);color:var(--c2c-fg);border:1px solid var(--c2c-border);
         border-radius:10px;padding:3px 10px;cursor:pointer;font-size:11px;
         font-family:ui-sans-serif,system-ui,sans-serif;`;
    btn.onmouseenter = () => btn.style.borderColor = "var(--c2c-mauve)";
    btn.onmouseleave = () => btn.style.borderColor = "var(--c2c-border)";
    btn.onclick = openPicker;
    if (window.__c2cTopDock) {
        window.__c2cTopDock.register(btn, { side: "left", order: 10 });
    } else {
        // Fallback: pin to a safe fixed position; retry dock registration
        // each animation frame in case _c2c_top_dock.js loads after us.
        btn.style.position = "fixed";
        btn.style.top = "60px";
        btn.style.left = "12px";
        btn.style.setProperty("z-index", "var(--c2c-z-dock, 2500)");
        document.body.appendChild(btn);
        const tryRegister = () => {
            if (window.__c2cTopDock) {
                btn.style.position = "";
                btn.style.top = "";
                btn.style.left = "";
                btn.style.removeProperty("z-index");
                window.__c2cTopDock.register(btn, { side: "left", order: 10 });
            } else {
                requestAnimationFrame(tryRegister);
            }
        };
        requestAnimationFrame(tryRegister);
    }
}

// ============================================================ event hookups
function _throttle(label) {
    const now = Date.now();
    if (now - _lastSnapAt < THROTTLE_MS) return;
    snapshot(label);
}

app.registerExtension({
    name: "c2c.autocheckpoint",
    settings: [
        { id: SETTING_ENABLED, name: "C2C ▸ Auto-Checkpoint ▸ Enabled",
          type: "boolean", default: true },
        { id: SETTING_RING, name: "C2C ▸ Auto-Checkpoint ▸ Ring size",
          type: "slider", attrs: { min: 5, max: 100, step: 1 }, default: RING_MAX_DEFAULT },
        { id: SETTING_BTN, name: "C2C ▸ Auto-Checkpoint ▸ Show top-left button",
          type: "boolean", default: true, onChange: ensureButton },
    ],
    commands: [
        { id: "c2c.autockpt.open",      label: "C2C: Open auto-checkpoint history", function: openPicker },
        { id: "c2c.autockpt.snapNow",   label: "C2C: Snapshot workflow now",        function: () => snapshot("manual") },
    ],
    keybindings: [
        { combo: { key: "h", ctrl: true, alt: true }, commandId: "c2c.autockpt.open" },
    ],
    async setup() {
        ensureButton();
        // before queue
        const orig = app.queuePrompt?.bind(app);
        if (orig) {
            app.queuePrompt = async function (...args) {
                // FIX: Delay snapshot slightly so it doesn't block the prompt from sending UI
                setTimeout(() => snapshot("auto: before queue"), 50);
                return orig(...args);
            };
        }
        // on graph mutations
        const orig2 = app.graph?.onNodeAdded;
        if (app.graph) {
            app.graph.onNodeAdded = function (node) {
                _throttle("auto: node added");
                return orig2?.call(this, node);
            };
            const orig3 = app.graph.onNodeRemoved;
            app.graph.onNodeRemoved = function (node) {
                _throttle("auto: node removed");
                return orig3?.call(this, node);
            };
        }
    },
});

window.__C2C_AUTOCKPT__ = { snapshot, openPicker, dbAll, dbClear };