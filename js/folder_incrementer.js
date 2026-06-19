import { app } from "../../scripts/app.js";

/**
 * FolderIncrementer JS companion
 * - Accepts any link type on the "trigger" / "trigger_image" / "trigger_video" inputs
 * - Auto-extracts filename from the connected loader node (LoadImage,
 *   LoadVideo, VHS_LoadVideo, etc.) and writes it into the
 *   source_filename widget (basename only — extension goes to source_extension).
 * - Traverses through Set/Get bus nodes, Reroute nodes, and arbitrary
 *   graph topologies to find the original source
 * - Honours the source_choice widget: "image" → trace trigger_image,
 *   "video" → trace trigger_video, "auto" → prefer video if connected,
 *   else image, else legacy `trigger`.
 */
app.registerExtension({
    name: "Comfy.FolderIncrementer",

    beforeRegisterNodeDef(nodeType, nodeData, app) {
        const targetNodes = [
            "FolderIncrementer",
            "FolderIncrementerReset",
            "FolderIncrementerSet",
        ];
        if (!targetNodes.includes(nodeData.name)) return;

        // Accept any type on trigger inputs
        nodeType.prototype.onConnectInput = function () { return true; };

        if (nodeData.name === "FolderIncrementer") {
            const _origVisible = nodeType.prototype.isWidgetVisible;
            nodeType.prototype.isWidgetVisible = function (widget) {
                if (widget?.name === "source_extension") return false;
                return _origVisible ? _origVisible.call(this, widget) : true;
            };

            const _configure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                _configure?.apply(this, arguments);
                requestAnimationFrame(() => {
                    try { this._fiNormalizeSource?.(); this._fiSyncSource?.(); } catch (_) {}
                });
            };
        }
    },

    loadedGraphNode(node) {
        if (node.comfyClass !== "FolderIncrementer") return;
        if (node._fiSetupDone) {
            requestAnimationFrame(() => {
                try { node._fiNormalizeSource?.(); node._fiSyncSource?.(); } catch (_) {}
            });
        }
    },

    nodeCreated(node) {
        if (node.comfyClass !== "FolderIncrementer") return;
        if (node._fiSetupDone) return;
        node._fiSetupDone = true;

        // Subgraph-aware graph accessor: when this node lives inside a
        // SubgraphNode wrapper, all of its links and siblings are in
        // node.graph (the inner LGraph), NOT in app.graph (the root).
        // Every helper below uses G() instead of app.graph for that
        // reason.
        const G = () => node.graph || app.graph;

        // Widget names commonly used by ComfyUI loaders to hold the
        // filename of the file they read.  Order matters: most-specific
        // first.  LoadImage uses "image", LoadVideo uses "video",
        // VHS_LoadVideo uses "video" too, audio loaders use "audio".
        const FILENAME_WIDGETS = ["image", "video", "filename", "file", "audio", "url"];

        // ── Check a single node for a filename widget ────────────────
        function getFilenameFromNode(n) {
            if (!n?.widgets) return null;
            for (const wName of FILENAME_WIDGETS) {
                const w = n.widgets.find(w => w.name === wName);
                if (w?.value && typeof w.value === "string" && w.value.trim()) {
                    const v = w.value.trim();
                    if (v.includes(".")) return v;
                }
            }
            return null;
        }

        // ── Resolve a Get node → find matching Set node ──────────────
        //    Different bus implementations (kjnodes, rgthree, cg-use-everywhere,
        //    easy-use) name the key widget differently — try them all.
        const BUS_KEY_WIDGETS = [
            "Constant", "constant", "value",
            "Key", "key", "Name", "name",
            "variable", "Variable", "label", "Label",
            "id", "ID",
        ];
        function getBusKey(n) {
            // Try every known widget name for the key.
            const w = n.widgets?.find(w => BUS_KEY_WIDGETS.includes(w.name));
            if (w?.value) return String(w.value).trim().toLowerCase();
            // Fall back to title prefix: "Set foo" / "Get_foo" / "Get: foo".
            const m = (n.title || "").match(/^(set|get)[\s_:\-]+(.+)$/i);
            return m ? m[2].trim().toLowerCase() : "";
        }

        function resolveGetNode(getNode) {
            const busName = getBusKey(getNode);
            if (!busName) return [];

            const results = [];
            const allNodes = G()._nodes || G().nodes || [];
            for (const n of allNodes) {
                if (n.id === getNode.id) continue;
                const title = (n.title || "").toLowerCase();
                const cls = (n.comfyClass || "").toLowerCase();
                const isSet = title.startsWith("set") || cls.startsWith("set")
                           || cls.includes("setnode");
                if (!isSet) continue;
                if (getBusKey(n) === busName) {
                    results.push(n);
                }
            }
            return results;
        }

        // ── Node types that are transparent routing (follow through) ──
        const ROUTING_RE = /reroute|universalreroute/i;
        function isRoutingNode(n) {
            const cls = n.comfyClass || "";
            return ROUTING_RE.test(cls);
        }

        function isGetBusNode(n) {
            const cls = (n.comfyClass || "").toLowerCase();
            const title = (n.title || "").toLowerCase();
            return title.startsWith("get") || cls.startsWith("get")
                || cls.includes("getnode");
        }

        // Follow first connected input of a node (upstream one hop)
        function followFirstInput(n) {
            if (!n?.inputs) return null;
            for (const inp of n.inputs) {
                if (inp.link == null) continue;
                const link = G().links[inp.link];
                if (!link) continue;
                return G().getNodeById(link.origin_id) || null;
            }
            return null;
        }

        // ── Input loader detection ──────────────────────────────────
        //    Loaders are exactly the nodes whose filename we DO want.
        //    We don't block them — we extract from them.
        const INPUT_LOADER_TYPES = [
            "LoadImage", "Load Image", "LoadImageMask",
            "LoadVideo", "Load Video", "VHS_LoadVideo", "VHS_LoadVideoPath",
            "LoadAudio", "Load Audio", "VHS_LoadAudio",
            "LoadImageBatch", "LoadImagesFromDir", "LoadImagesFromDirectory",
        ];

        function isInputLoader(n) {
            const cls = n.comfyClass || "";
            const title = n.title || "";
            return INPUT_LOADER_TYPES.some(t => cls.includes(t) || title.includes(t));
        }

        // ── BFS chain traversal to find a filename ───────────────────
        //    Walks upstream from a starting node, fanning out across
        //    every connected input of every intermediate node. Follows
        //    reroutes and Get→Set bus pairs transparently. Stops a
        //    branch when it hits a loader (whether or not it has a
        //    filename) so we don't escape the source island.
        //
        //    Returns the first filename found (BFS ⇒ closest in graph
        //    distance to ``startNode``), or null.
        function findFilenameFromChain(startNode, maxNodes = 200) {
            if (!startNode) return null;
            const visited = new Set();
            const queue = [startNode];

            while (queue.length && visited.size < maxNodes) {
                const current = queue.shift();
                if (!current || visited.has(current.id)) continue;
                visited.add(current.id);

                // Get bus → resolve to matching Set node(s), enqueue them
                if (isGetBusNode(current)) {
                    for (const setNode of resolveGetNode(current)) {
                        if (!visited.has(setNode.id)) queue.push(setNode);
                    }
                    continue;
                }

                // Routing/reroute → just follow through
                if (isRoutingNode(current)) {
                    const up = followFirstInput(current);
                    if (up) queue.push(up);
                    continue;
                }

                // Filename present on this node? Done.
                const fn = getFilenameFromNode(current);
                if (fn) return fn;

                // Loader with no filename widget → don't escape past it
                if (isInputLoader(current)) continue;

                // Generic processing node: enqueue every connected input
                if (current.inputs) {
                    for (const inp of current.inputs) {
                        if (inp.link == null) continue;
                        const link = G().links[inp.link];
                        if (!link) continue;
                        const upstream = G().getNodeById(link.origin_id);
                        if (upstream && !visited.has(upstream.id)) {
                            queue.push(upstream);
                        }
                    }
                }
            }
            return null;
        }

        // ── Resolve which input(s) to traverse ───────────────────────
        function findInput(name) {
            return node.inputs?.find(i => i.name === name) || null;
        }

        function getSourceNodeFromInput(inputName) {
            const inp = findInput(inputName);
            if (!inp || inp.link == null) return null;
            const linkInfo = G().links[inp.link];
            if (!linkInfo) return null;
            return G().getNodeById(linkInfo.origin_id) || null;
        }

        function getSourceChoice() {
            const w = node.widgets?.find(w => w.name === "source_choice");
            const v = (w?.value || "auto").toString().toLowerCase();
            // MANUAL bug-fix (Apr 2026): added 'custom' so users can
            // hand-type a name in the new custom_name widget instead
            // of being forced to wire up a loader trigger.
            return ["auto", "image", "video", "custom"].includes(v) ? v : "auto";
        }

        function getCustomName() {
            const w = node.widgets?.find(w => w.name === "custom_name");
            const v = (w?.value || "").toString().trim();
            return v;
        }

        // ── Name format (mirrors Python `_format_source_name`) ───────
        const NAME_FORMATS = ["basename", "strip_tags", "first_segment"];
        const TRAILING_TAG_RE = /[._\-](\d{3,4}p?|\d{2,3}fps|[248]k|uhd|hd|sd|sdr|hdr|raw|proxy|final|wip)$/i;

        function getNameFormat() {
            const w = node.widgets?.find(w => w.name === "name_format");
            const v = (w?.value || "basename").toString();
            return NAME_FORMATS.includes(v) ? v : "basename";
        }

        const KNOWN_EXT_RE = /\.(mp4|mov|webm|mkv|avi|m4v|flv|wmv|mpeg|mpg|ts|png|jpe?g|gif|webp|bmp|tiff?|tga|exr|hdr|heic|avif|wav|mp3|aac|flac|pdf|zip)$/i;

        function stripExt(filename) {
            if (!filename) return { stem: "", ext: "" };
            const m = String(filename).match(KNOWN_EXT_RE);
            if (m) {
                return { stem: filename.slice(0, -m[0].length), ext: m[0] };
            }
            return { stem: filename, ext: "" };
        }

        function splitFilename(filename) {
            return stripExt(filename);
        }

        function formatSourceName(rawFilename, fmt) {
            // rawFilename may include extension; status display always strips it.
            const { stem } = stripExt(rawFilename);
            let base = stem || rawFilename;
            if (!base) return rawFilename;
            if (fmt === "first_segment") {
                const m = base.split(/[._]/);
                return (m && m[0]) ? m[0] : base;
            }
            if (fmt === "strip_tags") {
                let cleaned = base;
                for (let i = 0; i < 4; i++) {
                    const next = cleaned.replace(TRAILING_TAG_RE, "");
                    if (next === cleaned) break;
                    cleaned = next;
                }
                return cleaned || base;
            }
            return base; // basename
        }

        function forceWidgetRefresh(w, value) {
            if (!w) return;
            const v = value !== undefined ? value : w.value;
            w.value = v;
            try { w.callback?.(v, app.canvas, node); } catch (_) {
                try { w.callback?.(v); } catch (_2) {}
            }
            if (w.inputEl) {
                w.inputEl.value = v;
                try {
                    w.inputEl.dispatchEvent(new Event("input", { bubbles: true }));
                    w.inputEl.dispatchEvent(new Event("change", { bubbles: true }));
                } catch (_) {}
            }
            try { node.onWidgetChanged?.(); } catch (_) {}
        }

        function writeSourceWidgets(fullFilename) {
            const sfWidget = node.widgets?.find(w => w.name === "source_filename");
            const extWidget = node.widgets?.find(w => w.name === "source_extension");
            const { stem, ext } = splitFilename(fullFilename || "");
            if (sfWidget) forceWidgetRefresh(sfWidget, stem);
            if (extWidget) forceWidgetRefresh(extWidget, ext);
            else if (ext) node._fiSourceExt = ext;
            return { stem, ext };
        }

        function normalizeSourceFilename() {
            const sfWidget = node.widgets?.find(w => w.name === "source_filename");
            if (!sfWidget?.value) return;
            const raw = String(sfWidget.value).trim();
            if (!raw || !raw.includes(".")) return;
            const { stem, ext } = splitFilename(raw);
            if (stem === raw) return;
            writeSourceWidgets(raw);
        }

        function hideExtensionWidget() {
            const extWidget = node.widgets?.find(w => w.name === "source_extension");
            if (!extWidget) return;
            extWidget.hidden = true;
            extWidget.type = "hidden";
            if (extWidget.computeSize) {
                extWidget.computeSize = () => [0, -4];
            }
        }

        // ── Extract filename honouring source_choice ─────────────────
        //    Tries the named trigger inputs first (in priority order
        //    based on source_choice). If none of them yields a result,
        //    falls back to scanning EVERY connected input on the node.
        //    Final fallback: scan the ENTIRE graph for loader nodes
        //    (so it works in big workflows even when nothing is wired
        //    into FolderIncrementer's trigger inputs).
        //
        //    Returns { filename, mode } where mode is one of
        //    "trigger", "input", "global".
        //
        //    BUG-FIX (Apr 2026): when source_choice is explicitly
        //    "image" or "video" we must NOT return a filename whose
        //    media type contradicts the user's choice. Previously, if
        //    source_choice="video" and only the legacy `trigger` input
        //    was connected to a still-image loader, we'd happily return
        //    the .png filename. Now we classify the candidate by
        //    extension (and by loader node type when available) and
        //    skip mismatches, falling through to the global type-aware
        //    scan instead.
        const IMAGE_EXT_RE = /\.(png|jpe?g|webp|bmp|tiff?|gif|tga|exr|hdr|heic|avif)$/i;
        const VIDEO_EXT_RE = /\.(mp4|mov|webm|mkv|avi|flv|m4v|wmv|mpeg|mpg|ts|gif)$/i;

        function classifyFilename(fn) {
            if (!fn) return "unknown";
            // .gif counts as video here only when explicitly chosen video;
            // default to image to match common usage.
            if (/\.gif$/i.test(fn)) return "image";
            if (VIDEO_EXT_RE.test(fn)) return "video";
            if (IMAGE_EXT_RE.test(fn)) return "image";
            return "unknown";
        }

        function matchesChoice(fn, choice) {
            if (choice === "auto") return true;
            const cls = classifyFilename(fn);
            if (cls === "unknown") return true; // can't tell -- be lenient
            return cls === choice;
        }

        function extractFilename() {
            const choice = getSourceChoice();

            // MANUAL bug-fix (Apr 2026): 'custom' short-circuits all
            // graph traversal. Whatever the user typed in custom_name
            // is the source name (verbatim, no auto-detection).
            if (choice === "custom") {
                const cn = getCustomName();
                if (cn) return { filename: cn, mode: "custom" };
                return null;
            }

            if (node.inputs) {
                const imgSrc    = getSourceNodeFromInput("trigger_image");
                const vidSrc    = getSourceNodeFromInput("trigger_video");
                const legacySrc = getSourceNodeFromInput("trigger");

                let order;
                if (choice === "image")      order = [imgSrc, legacySrc, vidSrc];
                else if (choice === "video") order = [vidSrc, legacySrc, imgSrc];
                else                         order = [vidSrc, imgSrc, legacySrc]; // auto

                // MANUAL bug-fix (Apr 2026): when the user explicitly
                // wires a loader into trigger_image / trigger_video /
                // trigger, that wire IS the explicit choice. Trust it
                // even if the resolved filename's extension does not
                // match `source_choice` (e.g. AV-Handles-trim of a
                // .mov is wired to trigger_image — the source name is
                // still the .mov, and the user knows it). The
                // classification gate now only applies to the global
                // fallback below, where we're guessing among unwired
                // candidates.
                for (const src of order) {
                    if (!src) continue;
                    const fn = findFilenameFromChain(src);
                    if (fn) {
                        return { filename: fn, mode: "trigger" };
                    }
                }

                // Fallback: scan every other connected input on the node.
                // Same trust principle: a real wire wins over a global guess.
                const namedInputs = new Set(["trigger", "trigger_image", "trigger_video"]);
                for (const inp of node.inputs) {
                    if (namedInputs.has(inp.name)) continue;
                    if (inp.link == null) continue;
                    const linkInfo = G().links[inp.link];
                    if (!linkInfo) continue;
                    const src = G().getNodeById(linkInfo.origin_id);
                    if (!src) continue;
                    const fn = findFilenameFromChain(src);
                    if (fn) {
                        return { filename: fn, mode: "input" };
                    }
                }
            }

            // ── Global fallback: scan every loader in the graph ──────
            const allNodes = G()._nodes || G().nodes || [];
            const candidates = [];
            for (const n of allNodes) {
                if (!isInputLoader(n)) continue;
                const fn = getFilenameFromNode(n);
                if (!fn) continue;
                const blob = ((n.comfyClass || "") + " " + (n.title || "")).toLowerCase();
                const isVideo = /video|vhs/.test(blob) || classifyFilename(fn) === "video";
                const isImage = (/image/.test(blob) || classifyFilename(fn) === "image") && !isVideo;
                candidates.push({ node: n, filename: fn, isVideo, isImage });
            }
            // BUG-FIX (Apr 2026): when explicit choice is set, drop
            // mismatched candidates entirely so we don't return the
            // wrong-type filename just because nothing of the right
            // type happens to be in the graph yet.
            let pool = candidates;
            if (choice === "video") pool = candidates.filter(c => c.isVideo);
            else if (choice === "image") pool = candidates.filter(c => c.isImage);
            if (pool.length === 0) return null;

            const score = (c) => {
                if (choice === "video") return c.isVideo ? 2 : (c.isImage ? 0 : 1);
                if (choice === "image") return c.isImage ? 2 : (c.isVideo ? 0 : 1);
                // auto: prefer video > image > other
                return c.isVideo ? 2 : (c.isImage ? 1 : 0);
            };
            // Single pass O(N): pick the highest-scoring candidate, tie-broken
            // by the largest node id (newest). Replaces three redundant full
            // sorts (two of which operated on `candidates`, whose result was
            // never used — the function returns the best of `pool`).
            let bestFilename = null, bestScore = -Infinity, bestId = -Infinity;
            for (const c of pool) {
                const s = score(c);
                const id = c.node.id || 0;
                if (s > bestScore || (s === bestScore && id > bestId)) {
                    bestFilename = c.filename; bestScore = s; bestId = id;
                }
            }
            return bestFilename ? { filename: bestFilename, mode: "global" } : null;
        }

        // ── Status display widget (read-only label on the node) ──────
        function ensureStatusWidget() {
            if (node._fiStatusWidget) return node._fiStatusWidget;
            // DOM widget = a small element shown on the node body
            if (typeof node.addDOMWidget === "function") {
                const el = document.createElement("div");
                el.style.cssText = [
                    "padding: 2px 6px",
                    "font: 11px monospace",
                    "color: var(--c2c-okPale)",
                    "background: var(--c2c-neutral950)",
                    "border: 1px solid var(--c2c-gray800)",
                    "border-radius: 3px",
                    "white-space: nowrap",
                    "overflow: hidden",
                    "text-overflow: ellipsis",
                    "min-height: 16px",
                ].join(";");
                el.title = "Source filename detected from upstream loader";
                el.textContent = "📄 (no source connected)";
                const w = node.addDOMWidget("source_status", "div", el, {
                    serialize: false,
                    getValue: () => el.textContent,
                    setValue: (v) => { el.textContent = v; },
                });
                w._el = el;
                node._fiStatusWidget = w;
                return w;
            }
            return null;
        }

        function setStatus(text) {
            const w = ensureStatusWidget();
            if (!w) return;
            const el = w._el;
            if (el && el.textContent !== text) {
                el.textContent = text;
                G().setDirtyCanvas(true);
            }
        }

        // ── Auto-fill source_filename + status display ───────────────
        function syncSourceFilename() {
            const result = extractFilename();
            const sfWidget = node.widgets?.find(w => w.name === "source_filename");
            if (result && result.filename) {
                // source_filename = stem only; extension → source_extension
                const { stem, ext } = writeSourceWidgets(result.filename);
                const fmt      = getNameFormat();
                const preview  = formatSourceName(stem, fmt);
                const sfx      = (node.widgets?.find(w => w.name === "suffix")?.value || "").trim();
                const sfxLabel = sfx ? `${sfx}` : "";
                const extLabel = ext ? ` ${ext}` : "";
                const tag = result.mode === "global" ? "\uD83C\uDF10"  // globe for global scan
                          : result.mode === "input"  ? "\uD83D\uDD0C"  // plug for non-trigger input
                          : result.mode === "custom" ? "\u270D\uFE0F"  // hand-writing for custom
                                                     : "\uD83D\uDCC4"; // page for trigger
                setStatus(`${tag} ${preview}${extLabel}${sfxLabel}`);
                G().setDirtyCanvas(true);
            } else {
                const manual = sfWidget?.value && sfWidget.value.trim();
                if (manual) {
                    // Legacy workflows may still have ext in source_filename — normalize once.
                    const { stem, ext } = splitFilename(manual);
                    if (stem !== manual) writeSourceWidgets(manual);
                    const fmt = getNameFormat();
                    const extW = node.widgets?.find(w => w.name === "source_extension");
                    const extShown = ext || extW?.value || "";
                    setStatus(`\uD83D\uDCDD ${formatSourceName(stem, fmt)}${extShown ? ` ${extShown}` : ""} (manual)`);
                } else {
                    setStatus("\uD83D\uDCC4 (no source connected)");
                }
            }
        }

        node._fiNormalizeSource = normalizeSourceFilename;
        node._fiSyncSource = syncSourceFilename;

        // Re-sync when source_filename edited manually — strip extension live
        const sfWidgetHook = node.widgets?.find(w => w.name === "source_filename");
        if (sfWidgetHook) {
            const origSfCb = sfWidgetHook.callback;
            sfWidgetHook.callback = function (v) {
                origSfCb?.apply(this, arguments);
                const raw = String(v ?? sfWidgetHook.value ?? "").trim();
                if (raw.includes(".")) {
                    writeSourceWidgets(raw);
                }
                setTimeout(syncSourceFilename, 0);
            };
        }

        const origOnConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function (type, index, connected, link_info) {
            origOnConnectionsChange?.apply(this, arguments);
            // Re-sync whether connecting OR disconnecting: a disconnect
            // may flip auto-mode from video back to image.
            setTimeout(syncSourceFilename, 150);
        };

        // Re-sync when source_choice widget changes
        const choiceWidget = node.widgets?.find(w => w.name === "source_choice");
        if (choiceWidget) {
            const origCb = choiceWidget.callback;
            choiceWidget.callback = function (v) {
                origCb?.apply(this, arguments);
                setTimeout(syncSourceFilename, 50);
            };
        }

        // Re-sync when name_format widget changes (live preview)
        const fmtWidget = node.widgets?.find(w => w.name === "name_format");
        if (fmtWidget) {
            const origCb = fmtWidget.callback;
            fmtWidget.callback = function (v) {
                origCb?.apply(this, arguments);
                setTimeout(syncSourceFilename, 0);
            };
        }

        // Re-sync when custom_name widget changes (so the status preview
        // updates live in 'custom' mode).
        const customWidget = node.widgets?.find(w => w.name === "custom_name");
        if (customWidget) {
            const origCb = customWidget.callback;
            customWidget.callback = function (v) {
                origCb?.apply(this, arguments);
                setTimeout(syncSourceFilename, 0);
            };
        }

        const origOnExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            origOnExecuted?.apply(this, arguments);
            syncSourceFilename();
        };

        // Sync before serialization (prompt queue) so Python gets fresh value
        const origOnSerialize = node.onSerialize;
        node.onSerialize = function (o) {
            normalizeSourceFilename();
            syncSourceFilename();
            origOnSerialize?.apply(this, arguments);
        };

        // Initial sync + periodic retry (graph may not be fully loaded yet)
        hideExtensionWidget();
        normalizeSourceFilename();
        setTimeout(() => { hideExtensionWidget(); normalizeSourceFilename(); syncSourceFilename(); }, 500);
        setTimeout(() => { hideExtensionWidget(); normalizeSourceFilename(); syncSourceFilename(); }, 2000);

        // Slow polling: re-scan every 3s so changes elsewhere in a big
        // workflow (e.g. user picks a different file in a Load Video
        // node, or loads a new Set/Get pair) propagate without needing
        // to wiggle our own connections.
        if (node._fiPollTimer) clearInterval(node._fiPollTimer);
        node._fiPollTimer = setInterval(() => {
            // Stop polling if the node is gone from the graph
            const g = G();
            const stillThere = (g._nodes || g.nodes || [])
                .some(n => n.id === node.id);
            if (!stillThere) {
                clearInterval(node._fiPollTimer);
                node._fiPollTimer = null;
                return;
            }
            syncSourceFilename();
        }, 3000);
    },
});
