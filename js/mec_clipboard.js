// MEC clipboard — Nuke-style portable copy/paste with full provenance.
//
// Captures EVERY widget value, plus the originating custom-node pack
// (python_module + display_name + best-effort GitHub URL discovered via
// ComfyUI-Manager's mapping endpoints). On paste:
//   1. Recreates nodes with positions, sizes, colours, properties, AND
//      widget->input conversions (the `inputs` array preserves which
//      widgets were converted to sockets). Done by replaying the same
//      serialize()/configure() format ComfyUI itself uses to save a
//      workflow, so anything that round-trips through Save/Load also
//      round-trips through MEC clipboard.
//   2. Re-wires intra-selection links.
//   3. Detects node types that don't exist on the target install,
//      gathers the unique missing packs, and shows a modal that lets
//      the user install them via ComfyUI-Manager (if installed) or
//      copy the GitHub clone URLs.
//
// Bindings (coexists with native LiteGraph clipboard):
//   * Ctrl+C / Ctrl+V on canvas               -> mirrored: native runs
//                                                AND we mirror enriched
//                                                JSON to OS clipboard.
//                                                On paste we prefer MEC
//                                                payload if present;
//                                                else native runs.
//   * Ctrl+Alt+C / Ctrl+Alt+V                  -> explicit MEC-only
//                                                (forces our path even
//                                                if native would also
//                                                fire).
//   * Canvas right-click & node right-click    -> menu entries.

import { app } from "../../scripts/app.js";

const CLIP_VERSION = "1.3";
const CLIP_HEADER  = `# MEC.clipboard ${CLIP_VERSION}`;
const HEADER_REGEX = /^#\s*MEC\.clipboard\s+\d+\.\d+/m;

// -----------------------------------------------------------------------
// Toast helper
// -----------------------------------------------------------------------
function _toast(msg, severity = "info", life = 3500) {
    try {
        app.extensionManager?.toast?.add({
            severity, summary: "MEC Clipboard", detail: msg, life,
        });
    } catch (_) { console.log("[MEC.clipboard]", msg); }
}

// -----------------------------------------------------------------------
// ComfyUI-Manager mapping cache
// -----------------------------------------------------------------------
// Manager exposes /customnode/getmappings?mode=nightly and
// /customnode/getlist?mode=local. Both let us map a custom-node pack
// folder name to its GitHub URL. Cache the result; both endpoints are
// slow and only needed at copy/paste time.
let _packMapCache = null;
let _packMapPromise = null;

async function _fetchPackMap() {
    if (_packMapCache) return _packMapCache;
    if (_packMapPromise) return _packMapPromise;
    _packMapPromise = (async () => {
        const out = {}; // packFolderName -> { url, title, class_types: Set }
        const tryFetch = async (url) => {
            try {
                const r = await fetch(url);
                if (!r.ok) return null;
                return await r.json();
            } catch (_) { return null; }
        };
        // Primary: /customnode/getmappings returns
        // { "<repo_url>": [["<class>", ...], { title_aux, ...}] }
        const map = await tryFetch("/customnode/getmappings?mode=nightly")
                 || await tryFetch("/customnode/getmappings?mode=cache");
        if (map && typeof map === "object") {
            for (const [repoUrl, entry] of Object.entries(map)) {
                if (!Array.isArray(entry) || entry.length < 1) continue;
                const classes = entry[0] || [];
                const meta = entry[1] || {};
                const folder = (repoUrl.split("/").pop() || "").replace(/\.git$/i, "");
                if (!folder) continue;
                out[folder] ||= { url: repoUrl, title: meta.title_aux || folder, class_types: new Set() };
                for (const c of classes) out[folder].class_types.add(c);
            }
        }
        // Augment with installed-pack list (mode=local) for packs not in the
        // public mapping.
        const list = await tryFetch("/customnode/getlist?mode=local");
        if (list && Array.isArray(list?.custom_nodes)) {
            for (const n of list.custom_nodes) {
                const url = n.reference || n.files?.[0];
                if (!url) continue;
                const folder = (url.split("/").pop() || "").replace(/\.git$/i, "");
                if (!folder) continue;
                out[folder] ||= { url, title: n.title || folder, class_types: new Set() };
                if (n.title && !out[folder].title) out[folder].title = n.title;
            }
        }
        _packMapCache = out;
        return out;
    })();
    return _packMapPromise;
}

// From a node instance, work out which custom_nodes folder it lives in
// (e.g. "ComfyUI-CustomNodePacks") and look up its repo URL.
function _packFolderForNode(node) {
    const nd = node.constructor?.nodeData || node?.nodeData;
    const pm = nd?.python_module || "";
    // python_module looks like "custom_nodes.ComfyUI-XYZ.nodes.foo" OR
    // "custom_nodes.ComfyUI-XYZ"; we want the segment right after
    // "custom_nodes."
    const m = /^custom_nodes\.([^.]+)/.exec(pm);
    return m ? m[1] : null;
}

async function _provenanceFor(node) {
    const folder = _packFolderForNode(node);
    if (!folder) return null;
    const map = await _fetchPackMap().catch(() => ({}));
    const e = map?.[folder];
    return {
        pack_folder: folder,
        pack_title: e?.title || folder,
        pack_url:   e?.url   || null,
        python_module: node.constructor?.nodeData?.python_module || null,
    };
}

// -----------------------------------------------------------------------
// Selection / serialisation
// -----------------------------------------------------------------------
function _selectedNodes() {
    const sel = app.canvas?.selected_nodes;
    if (!sel) return [];
    return Object.values(sel);
}

function _serializeNode(n, provenance) {
    // Use LiteGraph's own serialize() — the exact same format ComfyUI
    // writes when you Save the workflow. This preserves:
    //   - widgets_values (canonical array, in the order the node expects)
    //   - inputs[]  (incl. widget->input conversions: each converted
    //               input has a `widget: { name }` field)
    //   - outputs[], mode, order, type, size, pos, color, bgcolor,
    //     properties, flags, title.
    // Custom widgets (e.g. KJNodes color picker) implement their own
    // serializeValue(), which serialize() invokes — so `pad_color` ends
    // up as the string the node actually expects ("0, 0, 0"), not the
    // raw widget object.
    let core;
    try {
        core = n.serialize();
    } catch (e) {
        console.warn("[MEC.clipboard] node.serialize() failed:", e);
        core = {
            id: n.id, type: n.comfyClass || n.type,
            pos: n.pos, size: n.size,
            properties: n.properties, flags: n.flags,
        };
    }
    return {
        ...core,
        // Mirror our explicit fields on top — harmless duplicates that
        // make the JSON readable and let older paste code keep working.
        type: n.comfyClass || core.type,
        title: n.title,
        provenance: provenance || null,
    };
}

async function _gatherSubgraph(nodes) {
    const ids = new Set(nodes.map((n) => n.id));
    const out_nodes = [];
    for (const n of nodes) {
        const prov = await _provenanceFor(n);
        out_nodes.push(_serializeNode(n, prov));
    }
    const out_links = [];
    const links = app.graph?.links || {};
    for (const k in links) {
        const l = links[k];
        if (!l) continue;
        if (ids.has(l.origin_id) && ids.has(l.target_id)) {
            out_links.push({
                src_id: l.origin_id, src_slot: l.origin_slot,
                dst_id: l.target_id, dst_slot: l.target_slot,
                type: l.type,
            });
        }
    }
    const packs = {};
    for (const n of out_nodes) {
        const p = n.provenance;
        if (!p?.pack_folder) continue;
        packs[p.pack_folder] = {
            pack_folder: p.pack_folder,
            pack_title: p.pack_title,
            pack_url: p.pack_url,
        };
    }
    return {
        version: CLIP_VERSION,
        created: new Date().toISOString(),
        nodes: out_nodes,
        links: out_links,
        packs: Object.values(packs),
    };
}

// Synchronous variant used by the `copy` event hook (which cannot await).
// Uses only the already-warmed `_packMapCache`; if the cache hasn't loaded
// yet the provenance.pack_url comes back null but everything else is intact.
function _provenanceForSync(node) {
    const folder = _packFolderForNode(node);
    if (!folder) return null;
    const e = _packMapCache?.[folder];
    return {
        pack_folder: folder,
        pack_title: e?.title || folder,
        pack_url:   e?.url   || null,
        python_module: node.constructor?.nodeData?.python_module || null,
    };
}

function _gatherSubgraphSync(nodes) {
    const ids = new Set(nodes.map((n) => n.id));
    const out_nodes = [];
    for (const n of nodes) {
        out_nodes.push(_serializeNode(n, _provenanceForSync(n)));
    }
    const out_links = [];
    const links = app.graph?.links || {};
    for (const k in links) {
        const l = links[k];
        if (!l) continue;
        if (ids.has(l.origin_id) && ids.has(l.target_id)) {
            out_links.push({
                src_id: l.origin_id, src_slot: l.origin_slot,
                dst_id: l.target_id, dst_slot: l.target_slot,
                type: l.type,
            });
        }
    }
    const packs = {};
    for (const n of out_nodes) {
        const p = n.provenance;
        if (!p?.pack_folder) continue;
        packs[p.pack_folder] = {
            pack_folder: p.pack_folder,
            pack_title: p.pack_title,
            pack_url: p.pack_url,
        };
    }
    return {
        version: CLIP_VERSION,
        created: new Date().toISOString(),
        nodes: out_nodes,
        links: out_links,
        packs: Object.values(packs),
    };
}
async function _copy(specificNode = null) {
    const nodes = specificNode ? [specificNode] : _selectedNodes();
    if (!nodes.length) { _toast("Nothing selected", "warn"); return; }
    const payload = await _gatherSubgraph(nodes);
    const text = CLIP_HEADER + "\n" + JSON.stringify(payload, null, 2);
    try {
        await navigator.clipboard.writeText(text);
        const packs = payload.packs.length;
        // Push into the diagnostics sidebar history (kept tab-local).
        try {
            window.__MEC_CLIPBOARD_HISTORY__ = window.__MEC_CLIPBOARD_HISTORY__ || [];
            window.__MEC_CLIPBOARD_HISTORY__.push({
                ts: Date.now(),
                payload,
                text,
            });
            if (window.__MEC_CLIPBOARD_HISTORY__.length > 50) {
                window.__MEC_CLIPBOARD_HISTORY__.shift();
            }
        } catch (_) { /* ignore */ }
        _toast(
            `Copied ${nodes.length} node(s)` +
            (packs ? ` from ${packs} pack(s)` : "") +
            ` with full metadata`,
            "success",
        );
    } catch (e) {
        _toast("Clipboard write denied: " + (e?.message || e), "error");
    }
}

// -----------------------------------------------------------------------
// Missing-pack modal (Doctor-style "actionable next step")
// -----------------------------------------------------------------------
function _missingPackModal(missing /* [{folder,title,url,types:[]}] */) {
    return new Promise((resolve) => {
        const back = document.createElement("div");
        back.style.cssText = `
            position:fixed;inset:0;z-index:99999;background:#000a;
            display:flex;align-items:center;justify-content:center;
            font-family:system-ui,sans-serif;font-size:13px;color:#cdd6f4;
        `;
        const card = document.createElement("div");
        card.style.cssText = `
            background:#1e1e2e;border:1px solid #45475a;border-radius:8px;
            box-shadow:0 12px 40px #000a;padding:20px;max-width:640px;
            width:90%;max-height:80vh;overflow:auto;
        `;
        const h = document.createElement("h3");
        h.textContent = "Missing custom-node pack(s)";
        h.style.cssText = "margin:0 0 8px 0;color:#f38ba8;";
        card.appendChild(h);
        const sub = document.createElement("p");
        sub.innerHTML = `The clipboard payload references node types that aren't installed on this ComfyUI. ` +
                        `Install the missing packs and reload, then paste again.`;
        sub.style.cssText = "margin:0 0 12px 0;color:#bac2de;";
        card.appendChild(sub);

        for (const m of missing) {
            const row = document.createElement("div");
            row.style.cssText = `
                border:1px solid #313244;border-radius:6px;padding:10px;
                margin-bottom:8px;background:#181825;
            `;
            const t = document.createElement("div");
            t.innerHTML = `<b>${m.title || m.folder || "(unknown)"}</b>` +
                          (m.url
                              ? ` &mdash; <a href="${m.url}" target="_blank" style="color:#89b4fa;">${m.url}</a>`
                              : ` <span style="color:#f38ba8;">(no GitHub URL discovered \u2014 ComfyUI-Manager mapping unavailable)</span>`);
            t.style.cssText = "margin-bottom:4px;";
            row.appendChild(t);
            const types = document.createElement("div");
            types.textContent = `Needed types: ${m.types.join(", ")}`;
            types.style.cssText = "color:#a6adc8;font-size:11px;margin-bottom:6px;";
            row.appendChild(types);
            const btnRow = document.createElement("div");
            btnRow.style.cssText = "display:flex;gap:6px;flex-wrap:wrap;";
            if (m.url) {
                const cloneBtn = document.createElement("button");
                cloneBtn.textContent = "Copy clone command";
                cloneBtn.style.cssText = "padding:4px 10px;border-radius:4px;border:1px solid #45475a;background:#313244;color:#cdd6f4;cursor:pointer;";
                cloneBtn.onclick = () => {
                    navigator.clipboard.writeText(`git clone ${m.url} ComfyUI/custom_nodes/${m.folder}`);
                    _toast(`Clone command for ${m.folder} copied`, "success");
                };
                btnRow.appendChild(cloneBtn);

                const installBtn = document.createElement("button");
                installBtn.textContent = "Install via Manager";
                installBtn.style.cssText = "padding:4px 10px;border-radius:4px;border:1px solid #45475a;background:#a6e3a1;color:#11111b;cursor:pointer;font-weight:600;";
                installBtn.onclick = async () => {
                    installBtn.disabled = true; installBtn.textContent = "Installing\u2026";
                    const ok = await _managerInstall(m);
                    installBtn.textContent = ok ? "Queued \u2713 \u2014 restart ComfyUI" : "Failed (see console)";
                    installBtn.style.background = ok ? "#94e2d5" : "#f38ba8";
                };
                btnRow.appendChild(installBtn);
            }
            row.appendChild(btnRow);
            card.appendChild(row);
        }

        const footer = document.createElement("div");
        footer.style.cssText = "display:flex;justify-content:flex-end;gap:8px;margin-top:8px;";
        const closeBtn = document.createElement("button");
        closeBtn.textContent = "Close";
        closeBtn.style.cssText = "padding:5px 14px;border-radius:4px;border:1px solid #45475a;background:#313244;color:#cdd6f4;cursor:pointer;";
        closeBtn.onclick = () => { document.body.removeChild(back); resolve(false); };
        footer.appendChild(closeBtn);
        const proceedBtn = document.createElement("button");
        proceedBtn.textContent = "Paste anyway (skip missing)";
        proceedBtn.style.cssText = "padding:5px 14px;border-radius:4px;border:1px solid #45475a;background:#fab387;color:#11111b;cursor:pointer;";
        proceedBtn.onclick = () => { document.body.removeChild(back); resolve(true); };
        footer.appendChild(proceedBtn);
        card.appendChild(footer);

        back.appendChild(card);
        document.body.appendChild(back);
    });
}

async function _managerInstall(pack) {
    // Tries the modern ComfyUI-Manager install endpoint first, then the
    // older one. Both expect a custom-node descriptor.
    const body = {
        id: pack.folder,
        title: pack.title || pack.folder,
        reference: pack.url,
        files: [pack.url],
        install_type: "git-clone",
        repository: pack.url,
    };
    const tryPost = async (url) => {
        try {
            const r = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            return r.ok;
        } catch (_) { return false; }
    };
    if (await tryPost("/manager/queue/install")) return true;
    if (await tryPost("/customnode/install"))    return true;
    console.warn("[MEC.clipboard] Manager install endpoints failed; user must clone manually:", pack.url);
    return false;
}

// -----------------------------------------------------------------------
// Paste
// -----------------------------------------------------------------------
function _findHeader(text) {
    if (!text || !HEADER_REGEX.test(text)) return null;
    const j = text.indexOf("{");
    if (j < 0) return null;
    try { return JSON.parse(text.slice(j)); } catch (_) { return null; }
}

async function _paste() {
    let text = "";
    try { text = await navigator.clipboard.readText(); }
    catch (e) { _toast("Clipboard read denied (browser permission)", "error"); return; }
    return _pasteFromText(text);
}

async function _pasteFromText(text) {
    const data = _findHeader(text);
    if (!data || !Array.isArray(data.nodes)) {
        _toast("Clipboard does not contain MEC payload", "warn"); return;
    }

    // -- Missing-type detection ------------------------------------------------
    const reg = LiteGraph.registered_node_types || {};
    const missingByPack = {};
    const orphans = [];
    for (const nd of data.nodes) {
        if (reg[nd.type]) continue;
        const p = nd.provenance;
        if (p?.pack_folder) {
            const k = p.pack_folder;
            missingByPack[k] ||= {
                folder: p.pack_folder,
                title:  p.pack_title || p.pack_folder,
                url:    p.pack_url   || null,
                types:  [],
            };
            if (!missingByPack[k].types.includes(nd.type)) missingByPack[k].types.push(nd.type);
        } else {
            orphans.push(nd.type);
        }
    }
    const missing = Object.values(missingByPack);
    if (orphans.length) {
        missing.push({ folder: "(unknown)", title: "Unknown pack", url: null, types: [...new Set(orphans)] });
    }

    if (missing.length) {
        const proceed = await _missingPackModal(missing);
        if (!proceed) return;
    }

    // -- Instantiate -----------------------------------------------------------
    const cursor = app.canvas?.graph_mouse || [0, 0];
    let originX = null, originY = null;
    const idMap = {};
    let skipped = 0;
    for (const nd of data.nodes) {
        const node = LiteGraph.createNode(nd.type);
        if (!node) { skipped++; continue; }

        // Add to graph FIRST so the node has a real id and ComfyUI can
        // run its onNodeCreated hooks (which create widgets, sockets,
        // and any pack-specific scaffolding the configure() step relies
        // on).
        app.graph.add(node);

        // Compute paste anchor offset.
        const srcPos = Array.isArray(nd.pos) ? nd.pos : [0, 0];
        if (originX === null) { originX = srcPos[0]; originY = srcPos[1]; }
        const dstPos = [
            cursor[0] + (srcPos[0] - originX),
            cursor[1] + (srcPos[1] - originY),
        ];

        // configure() is LiteGraph's own load-from-workflow method. It
        // applies widgets_values (the canonical ordered array), the
        // inputs[] array (preserving widget->input conversions like the
        // blue dots beside `width`/`height` on KJNodes' Resize), the
        // outputs[] array, properties, flags, mode, and size. We strip
        // id/pos so the new node gets a fresh id and our paste anchor.
        const cfg = { ...nd };
        delete cfg.id;
        delete cfg.provenance;     // not part of LGraph schema
        delete cfg.pos;
        try {
            if (typeof node.configure === "function") {
                node.configure(cfg);
            } else {
                // Defensive fallback (shouldn't trigger in modern ComfyUI).
                if (Array.isArray(cfg.widgets_values) && node.widgets) {
                    cfg.widgets_values.forEach((v, i) => {
                        const w = node.widgets[i];
                        if (!w) return;
                        try { w.value = v; w.callback?.(v, app.canvas, node); }
                        catch (_) {}
                    });
                }
            }
        } catch (err) {
            console.warn("[MEC.clipboard] configure() failed for", nd.type, err);
        }

        // Apply paste anchor + cosmetic fields AFTER configure so they
        // win over any defaults configure() might restore.
        node.pos = dstPos;
        if (nd.title) node.title = nd.title;
        if (nd.color) node.color = nd.color;
        if (nd.bgcolor) node.bgcolor = nd.bgcolor;
        if (Array.isArray(nd.size)) node.size = [nd.size[0], nd.size[1]];

        idMap[nd.id] = node;
    }
    for (const l of data.links || []) {
        const src = idMap[l.src_id], dst = idMap[l.dst_id];
        if (!src || !dst) continue;
        try { src.connect(l.src_slot, dst, l.dst_slot); }
        catch (err) { console.warn("[MEC.clipboard] link failed", err); }
    }
    try {
        app.canvas.deselectAllNodes?.();
        for (const n of Object.values(idMap)) app.canvas.selectNode?.(n, true);
    } catch (_) {}
    app.graph.setDirtyCanvas(true, true);
    _toast(
        `Pasted ${Object.keys(idMap).length} node(s)` +
        (skipped ? ` (${skipped} skipped \u2014 missing pack)` : ""),
        skipped ? "warn" : "success",
    );
}

// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// 3-tier wiring (clipboard events + keyboard + canvas menu + node menu)
// -----------------------------------------------------------------------
//
// Native LiteGraph stores its own clipboard inside `localStorage` and
// listens for keydown to copy/paste from there. The OS clipboard
// (`navigator.clipboard`) is a separate channel.
//
// We hook the *clipboard* DOM events (`copy` / `paste`) at the document
// level, capture phase — these fire synchronously when Ctrl+C / Ctrl+V
// (or right-click menu Copy/Paste) is triggered. Doing it here means:
//
//   * Ctrl+C  -> `copy` event fires; we synchronously write MEC payload
//                into `e.clipboardData`. Native LiteGraph's keydown
//                listener still runs and stores its localStorage copy
//                so same-tab paste continues to work normally.
//   * Ctrl+V  -> `paste` event fires; we read `e.clipboardData`. If it
//                carries an MEC header we run our paste path and
//                preventDefault so native doesn't double-paste.
//                Otherwise we let native handle it.
//   * Ctrl+Alt+C / Ctrl+Alt+V -> kept as keyboard fallback (always MEC).
//
function _eventInTextField(target) {
    return !!(target && (
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.isContentEditable
    ));
}

function _focusOnCanvas() {
    const a = document.activeElement;
    if (!a || a === document.body || a === document.documentElement) return true;
    if (a.tagName === "CANVAS") return true;
    return false;
}

// ------ COPY -----------------------------------------------------------
document.addEventListener("copy", (ev) => {
    try {
        if (_eventInTextField(ev.target)) return;            // typing → native
        if (!_focusOnCanvas()) return;                       // not on graph → native
        const nodes = _selectedNodes();
        if (!nodes.length) return;                           // nothing → native (selection in textareas etc.)
        // Build payload SYNCHRONOUSLY using whatever pack info is already cached.
        // _gatherSubgraph is async only because of the network pack-map fetch;
        // the pack-map is pre-warmed in setup() so the subsequent calls return
        // from cache. We still must do the actual work synchronously here, so
        // call the synchronous variant.
        const payload = _gatherSubgraphSync(nodes);
        const text = CLIP_HEADER + "\n" + JSON.stringify(payload, null, 2);
        ev.clipboardData.setData("text/plain", text);
        ev.preventDefault();                                 // we own the OS clipboard
        // Sidebar history.
        try {
            window.__MEC_CLIPBOARD_HISTORY__ = window.__MEC_CLIPBOARD_HISTORY__ || [];
            window.__MEC_CLIPBOARD_HISTORY__.push({ ts: Date.now(), payload, text });
            if (window.__MEC_CLIPBOARD_HISTORY__.length > 50) {
                window.__MEC_CLIPBOARD_HISTORY__.shift();
            }
        } catch (_) { /* ignore */ }
        const packs = payload.packs.length;
        _toast(
            `Copied ${nodes.length} node(s)` +
            (packs ? ` from ${packs} pack(s)` : "") +
            ` with full metadata`,
            "success",
        );
    } catch (e) {
        console.warn("[MEC.clipboard] copy hook failed:", e);
        // Don't preventDefault → native copy still runs.
    }
}, true);

// ------ PASTE ----------------------------------------------------------
document.addEventListener("paste", (ev) => {
    try {
        if (_eventInTextField(ev.target)) return;            // pasting into a field → native
        if (!_focusOnCanvas()) return;
        const text = ev.clipboardData?.getData?.("text/plain") || "";
        if (!HEADER_REGEX.test(text)) return;                // not ours → native
        ev.preventDefault();
        ev.stopPropagation();
        _pasteFromText(text);
    } catch (e) {
        console.warn("[MEC.clipboard] paste hook failed:", e);
    }
}, true);

// ------ Ctrl+Alt+C / Ctrl+Alt+V keyboard fallback ----------------------
window.addEventListener("keydown", (ev) => {
    if (!(ev.ctrlKey || ev.metaKey) || !ev.altKey || ev.shiftKey) return;
    if (_eventInTextField(ev.target)) return;
    const k = ev.key.toLowerCase();
    if (k === "c") { ev.preventDefault(); _copy(); }
    else if (k === "v") { ev.preventDefault(); _paste(); }
}, true);

app.registerExtension({
    name: "MEC.Clipboard",
    async setup() {
        console.log("[MEC.clipboard] v" + CLIP_VERSION + " loaded \u2014 native Ctrl+C/Ctrl+V via clipboard events, Ctrl+Alt+C/V fallback");
        // Pre-warm the pack map (non-blocking).
        _fetchPackMap().catch(() => {});

        const origMenu = LGraphCanvas.prototype.getCanvasMenuOptions;
        LGraphCanvas.prototype.getCanvasMenuOptions = function () {
            const opts = origMenu.apply(this, arguments) || [];
            opts.push(null);
            opts.push({ content: "MEC: Copy node(s) with metadata (Ctrl+C / Ctrl+Alt+C)", callback: () => _copy() });
            opts.push({ content: "MEC: Paste node(s) from clipboard (Ctrl+V / Ctrl+Alt+V)", callback: _paste });
            return opts;
        };
        const origNodeMenu = LGraphCanvas.prototype.getNodeMenuOptions;
        LGraphCanvas.prototype.getNodeMenuOptions = function (node) {
            const opts = origNodeMenu.apply(this, arguments) || [];
            opts.push(null);
            opts.push({ content: "MEC: Copy with metadata", callback: () => {
                try { app.canvas.selectNode?.(node, true); } catch (_) {}
                _copy(node);
            }});
            return opts;
        };
    },
});
