import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { vueSyncNodeWidgets } from "./_widget_visibility.js";
import { sourceFromWidgets, sourceFromValue, kindRank, KNOWN_EXT_RE as SOURCE_EXT_RE } from "./_fi_source_name.js";

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
// ── Resync on CHANGE, never on a timer ──────────────────────────────────────
// Each node used to poll every 3 s: a walk of the whole graph (every node,
// every widget) per incrementer, forever, including in background tabs. On a
// big production graph with several incrementers that is steady main-thread
// load for nothing. The front-end already announces every real edit - node
// added or removed, link changed, a loader's file picked - as `graphChanged`
// (ChangeTracker, the same signal its autosave uses). Listen to that, once for
// the page, debounced; a hidden tab catches up when it becomes visible.
const _fiLive = new Set();
let _fiTimer = 0, _fiPendingHidden = false;
// ── Work once per graph change, not once per serialize (ledger L4.11) ───────
// The change tracker serializes the whole graph after nearly every action, and
// other extensions serialize too; onSerialize used to re-walk the graph for every
// incrementer each time - on the owner's graph collectFilenamesFromChain was ~85%
// of the main thread. Now every graph change bumps _fiVersion; a node re-syncs only
// when its last sync is older than that, and what all incrementers in a graph read
// (each node's source, loader detection, the Set/Get bus map, the global loader
// list) is computed once per version and shared.
let _fiVersion = 1;
const _fiCaches = new WeakMap();          // graph -> per-version shared lookups
function _fiBump() { _fiVersion++; }
function _fiResyncAll() {
    _fiTimer = 0;
    if (typeof document !== "undefined" && document.hidden) { _fiPendingHidden = true; return; }
    for (const n of _fiLive) {
        if (!n.graph) { _fiLive.delete(n); continue; }   // removed from its graph
        if (n._fiSyncedVersion === _fiVersion) continue;   // nothing changed since its last sync
        try { n._fiSyncSource?.(); } catch (_) { /* a name preview must never break the page */ }
    }
}
function _fiSchedule() {
    _fiBump();
    if (_fiTimer) clearTimeout(_fiTimer);
    _fiTimer = setTimeout(_fiResyncAll, 250);
}
if (!globalThis.__C2C_FI_EVENTS__) {
    globalThis.__C2C_FI_EVENTS__ = true;
    try { api.addEventListener("graphChanged", _fiSchedule); } catch (_) { /* old front-end */ }
    if (typeof document !== "undefined") {
        document.addEventListener("visibilitychange", () => {
            if (!document.hidden && _fiPendingHidden) { _fiPendingHidden = false; _fiSchedule(); }
        });
    }
}

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
        // VFX/OCIO nodes (ComfyUI-NukeMaxNodes LoadEXRMEC, ComfyUI-OCIO
        // OCIORead, generic file-path STRING inputs) name the path widget
        // "source", "file_path", "path", "exr_path" or "image_path" —
        // added so a FolderIncrementer wired downstream of an EXR/OCIO
        // read auto-detects the plate name instead of falling through to
        // the nearest ref-image loader.
        const FILENAME_WIDGETS = [
            "image", "video", "filename", "file", "audio", "url",
            "source", "file_path", "filepath", "path",
            "exr_path", "exr", "image_path", "sequence_path",
            // Found on live nodes during the 2026-09-22 sweep:
            // NukeMax_MochaImportShapesAsMaskPaste uses `uploaded_file`,
            // MiniMaxH3_DCCBridge uses `exr_sequence`, and several readers
            // name a directory rather than a file.
            "uploaded_file", "exr_sequence", "directory", "folder",
            "pattern", "sequence", "frames_path",
        ];

        // ── The source a single node reads ───────────────────────────
        //    Judged by VALUE, not by node type or a fixed widget list
        //    (_fi_source_name.js): a file, a folder, a glob or a sequence
        //    pattern in ANY widget. The old rule kept a value only if it
        //    contained a dot, so every folder loader - Load Images From Dir,
        //    VHS Load Images, WAS Load Image Batch, EXR sequence folders - was
        //    skipped and the name came from some other loader, or none.
        function readSourceFromNode(n) {
            try { return sourceFromWidgets(n?.widgets); } catch (_) { return null; }
        }
        // Shared by every incrementer in this graph until the next graph change:
        // reading `widgets` is not free on this front-end (subgraph nodes project
        // promoted widgets on every access), so each node is read once per version.
        function graphCache() {
            const g = G();
            let c = _fiCaches.get(g);
            if (!c || c.version !== _fiVersion) {
                c = { version: _fiVersion, src: new Map(), loader: new Map(), bus: null, candidates: null };
                _fiCaches.set(g, c);
            }
            return c;
        }
        function getSourceFromNode(n) {
            if (!n) return null;
            const c = graphCache();
            if (c.src.has(n)) return c.src.get(n);
            const s = readSourceFromNode(n);
            c.src.set(n, s);
            return s;
        }
        function getFilenameFromNode(n) {
            return getSourceFromNode(n)?.filename || null;
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

        // key -> Set nodes, built once per graph version: resolving a Get node used
        // to scan every node in the graph, once per Get node met during a walk.
        function busMap() {
            const c = graphCache();
            if (c.bus) return c.bus;
            const m = new Map();
            for (const n of (G()._nodes || G().nodes || [])) {
                const title = (n.title || "").toLowerCase();
                const cls = (n.comfyClass || "").toLowerCase();
                if (!(title.startsWith("set") || cls.startsWith("set") || cls.includes("setnode"))) continue;
                const key = getBusKey(n);
                if (!key) continue;
                if (!m.has(key)) m.set(key, []);
                m.get(key).push(n);
            }
            c.bus = m;
            return m;
        }

        function resolveGetNode(getNode) {
            const busName = getBusKey(getNode);
            if (!busName) return [];
            return (busMap().get(busName) || []).filter(n => n.id !== getNode.id);
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
        //    BUG FIX (2026-09-22): the VFX readers were missing entirely.
        //    FILENAME_WIDGETS already knew about `file_path`/`exr_path`, and
        //    the classifier already knew a numbered EXR sequence is a
        //    moving-image source — but an EXR/OCIO/Nuke read was not in THIS
        //    list, which does two jobs and failed both:
        //      * it stops the upstream walk, so an unrecognised reader was
        //        walked straight past and the search escaped into another
        //        island and found a ref-image loader there;
        //      * it drives the many-loaders disambiguator, so an EXR read
        //        could never win against a recognised LoadImage.
        //    Net effect: the right filename, taken from the wrong node —
        //    which is the reported "shows ref image, not the video/exr".
        const INPUT_LOADER_TYPES = [
            // ComfyUI core / VHS
            "LoadImage", "Load Image", "LoadImageMask",
            "LoadVideo", "Load Video", "VHS_LoadVideo", "VHS_LoadVideoPath",
            "LoadAudio", "Load Audio", "VHS_LoadAudio",
            "LoadImageBatch", "LoadImagesFromDir", "LoadImagesFromDirectory",
            // EXR / DPX / image-sequence readers (NukeMax, CNP, MiniMax)
            "LoadEXRMEC", "SaveEXRMEC",
            "NukeMax_EXRSequenceLoad", "NukeMax_VideoSequenceLoad",
            "NukeMax_EXRChannelRouter", "NukeMax_ReadMultiPass",
            "EXRMetadataReaderMEC", "MiniMaxH3_DCCBridge",
            // Mocha shape imports read a .mocha/.shape file and output a MASK,
            // so they are a source too. Found by the live-reader check rather
            // than by hand, which is the point of having it.
            "NukeMax_MochaImportShapesAsMask",
            "NukeMax_MochaImportShapesAsMaskPaste",
            // OCIO / colour-managed reads
            "OCIORead", "OCIO Read", "NukeMax_OCIOFileTransform",
            // Nuke-style reads, and the generic spellings third-party packs use
            "NukeRead", "Nuke Read", "ReadEXR", "Read EXR", "ReadImage",
            "ImageSequenceLoader", "LoadImageSequence", "LoadEXR",
            "LoadEXRSequence", "Load EXR", "ImageFromPath", "LoadImageFromPath",
            // Pixaroma loaders
            "PixaromaLoadVideo", "PixaromaLoadImage", "PixaromaLoadImagesFolder",
            "PixaromaLoadVideoFrame", "PixaromaLoadAudio",
        ];

        // Any node that READS media counts, whatever pack it is from: its
        // class or title says load/read/import, or it has no wired inputs at
        // all (a pure source) - and in both cases a widget names a file or
        // folder. The list above stays as the fast, certain path.
        const LOADER_WORD_RE = /(load|read|import|open|sequence|footage|plate|batch|from.?(dir|folder|path))/i;
        // Same "class or title contains one of these" test as before, as ONE regex:
        // the 60-name .some(includes) was the hottest line of a global scan.
        const LOADER_TYPES_RE = new RegExp(
            INPUT_LOADER_TYPES.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|"));
        function computeIsInputLoader(n) {
            const cls = n.comfyClass || "";
            const title = n.title || "";
            if (LOADER_TYPES_RE.test(cls) || LOADER_TYPES_RE.test(title)) return true;
            if (!getSourceFromNode(n)) return false;
            if (LOADER_WORD_RE.test(cls) || LOADER_WORD_RE.test(title)) return true;
            return !(n.inputs || []).some(i => i && i.link != null);
        }
        function isInputLoader(n) {
            const c = graphCache();
            if (c.loader.has(n)) return c.loader.get(n);
            const v = computeIsInputLoader(n);
            c.loader.set(n, v);
            return v;
        }

        // ── BFS chain traversal: COLLECT every filename in the source island ─
        //    Walks upstream from a starting node, fanning out across every
        //    connected input of every intermediate node. Follows reroutes and
        //    Get→Set bus pairs transparently. Stops a branch when it hits a
        //    loader (whether or not it has a filename) so we don't escape the
        //    source island.
        //
        //    Returns an array of { filename, cls } in BFS order (closest in
        //    graph distance to ``startNode`` first). cls ∈ image|video|unknown.
        //
        //    WHY collect-all (not return-first): a decode/sampler chain reaches
        //    BOTH a ref-image loader AND the real video loader. The ref image is
        //    usually FEWER hops away, so a return-first walk always picked the
        //    image — that is the "shows ref image name but not video name" bug,
        //    and the reason switching source_choice to 'video' felt "stuck"
        //    (the walk never consulted the choice). We now gather every
        //    candidate and let pickFilenameByChoice() honour source_choice.
        function collectFilenamesFromChain(startNode, maxNodes = 200) {
            const found = [];
            if (!startNode) return found;
            const seenFn = new Set();
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

                // Filename present on this node? Record it (dedup), then stop
                // this branch — but keep draining the queue so sibling branches
                // (e.g. the video branch alongside a ref-image branch) are seen.
                const src = getSourceFromNode(current);
                if (src) {
                    if (!seenFn.has(src.filename)) {
                        seenFn.add(src.filename);
                        found.push({ filename: src.filename, cls: src.kind });
                    }
                    continue;
                }

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
            return found;
        }

        // Pick the best filename from a collected chain, honouring source_choice.
        //   video → first video; else first unknown; else first found (closest).
        //   image → first image; else first unknown; else first found.
        //   auto  → prefer video, then image, then first found.
        // This is what makes "select video/image after connecting" actually
        // re-pick (was stuck), and a wired decode resolve to the VIDEO name.
        function pickFilenameByChoice(found, choice) {
            if (!found || !found.length) return null;
            // Best kind for the choice wins; among equals the CLOSEST (BFS
            // order) wins. A folder / numbered sequence is a moving-image
            // source, so it satisfies "video" and outranks a lone still.
            let best = null, bestScore = -Infinity;
            for (const f of found) {
                let sc = kindRank(f.cls, choice);
                if (choice === "exr" && isExrFile(f.filename)) sc += 2;
                if (sc > bestScore) { best = f; bestScore = sc; }
            }
            return best ? best.filename : null;
        }

        // Thin wrapper kept for the call sites below.
        function findFilenameFromChain(startNode, choice = "auto") {
            return pickFilenameByChoice(collectFilenamesFromChain(startNode), choice);
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
            // VFX (2026-08-13): added 'exr' — explicit EXR/DPX/CIN plate
            // choice. Works without a trigger (source_path STRING).
            return ["auto", "image", "video", "exr", "custom"].includes(v) ? v : "auto";
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

        const KNOWN_EXT_RE = SOURCE_EXT_RE;

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

        // VFX (2026-08-13): derive a display stem from a FULL path or
        // filename (e.g. ".../shot_v001/shot_v001.1001.exr" → "shot_v001").
        // Strips directory, extension, and the trailing frame token so a
        // sequence resolves to one stem. Used to show the detected plate
        // name in 'path mode' (source_path wired, no trigger) so the user
        // can see the node found the EXR without connecting a trigger.
        const SEQ_STRIP_EXT_RE = /\.(exr|dpx|cin|tiff?|tga|png|jpe?g|hdr)$/i;
        const SEQ_STRIP_FRAME_RE = /[._-](\d{2,8}|#{2,8}|%0?\d*d)$/;
        function stemFromPath(raw) {
            if (!raw) return "";
            const judged = sourceFromValue(String(raw), true);
            let v = judged ? judged.filename : String(raw).trim();
            const slash = Math.max(v.lastIndexOf("/"), v.lastIndexOf("\\"));
            if (slash >= 0 && slash < v.length - 1) v = v.slice(slash + 1);
            const { stem, ext } = splitFilename(v);
            let s = stem || v;
            if (ext && SEQ_STRIP_EXT_RE.test(ext)) {
                s = s.replace(SEQ_STRIP_FRAME_RE, "");
            }
            return s;
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

        // PERF: this used to write (and fire the callback) even when the value
        // was already right. source_filename's callback schedules a re-sync,
        // the re-sync writes the same stem again - an endless loop measured at
        // ~150 laps/s and ~700 ms of main thread per second, per workflow with
        // this node, at idle. Now: an unchanged value is left alone, and a
        // callback fired by our own write does not schedule another sync.
        function forceWidgetRefresh(w, value) {
            if (!w) return;
            const v = value !== undefined ? value : w.value;
            if (w.value === v && (!w.inputEl || w.inputEl.value === v)) return;
            w.value = v;
            node._fiWriting = true;
            try {
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
            } finally {
                node._fiWriting = false;
            }
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
            vueSyncNodeWidgets(node);
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
        const IMAGE_EXT_RE = /\.(png|jpe?g|webp|bmp|tiff?|gif|tga|exr|dpx|cin|hdr|heic|avif)$/i;
        const VIDEO_EXT_RE = /\.(mp4|mov|webm|mkv|avi|flv|m4v|wmv|mpeg|mpg|ts|gif)$/i;

        // A FRAME-SEQUENCE name: shot.1001.exr, shot_0042.exr, shot.####.exr,
        // shot.%04d.exr. In VFX the moving-image source IS a numbered EXR/DPX
        // sequence, so these must satisfy source_choice="video" even though
        // their extension is a still-image one.
        const SEQ_EXT_RE = /\.(exr|dpx|cin|tiff?|tga|png|jpe?g|hdr)$/i;
        const SEQ_FRAME_RE = /[._-](\d{2,8}|#{2,8}|%0?\d*d)(?=\.[A-Za-z0-9]+$)/;

        function isFrameSequence(fn) {
            return SEQ_EXT_RE.test(fn) && SEQ_FRAME_RE.test(fn);
        }

        // VFX (2026-08-13): an EXR/DPX/CIN plate — single file OR numbered
        // sequence. Used by source_choice='exr' to pick the plate
        // preferentially. Distinct from classifyFilename (which returns
        // 'video' for sequences) so the 'video' choice still picks exr
        // sequences while 'exr' picks exr/dpx/cin specifically.
        const EXR_FILE_RE = /\.(exr|dpx|cin)$/i;
        function isExrFile(fn) {
            return !!fn && EXR_FILE_RE.test(fn);
        }

        function classifyFilename(fn) {
            if (!fn) return "unknown";
            // BUG FIX (2026-08-01): an EXR sequence used as the moving-image
            // source classified as "image" (exr is in IMAGE_EXT_RE), so with
            // source_choice="video" the pick found no video, no unknown, and
            // fell through to found[0] — the CLOSEST node in the graph, which
            // when wired through a VAE decode is the ref_image. That is the
            // reported "takes ref_image name even with source_choice=video".
            // A numbered sequence is a moving-image source; classify it as one.
            if (isFrameSequence(fn)) return "video";
            // .gif counts as video here only when explicitly chosen video;
            // default to image to match common usage.
            if (/\.gif$/i.test(fn)) return "image";
            if (VIDEO_EXT_RE.test(fn)) return "video";
            if (IMAGE_EXT_RE.test(fn)) return "image";
            return "unknown";
        }

        function matchesChoice(fn, choice) {
            if (choice === "auto") return true;
            if (choice === "exr") {
                // Exr choice: accept exr/dpx/cin, or any sequence (classified
                // video), or unknown (can't tell — be lenient).
                return isExrFile(fn) || classifyFilename(fn) === "video"
                    || classifyFilename(fn) === "unknown";
            }
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
                    const fn = findFilenameFromChain(src, choice);
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
                    const fn = findFilenameFromChain(src, choice);
                    if (fn) {
                        return { filename: fn, mode: "input" };
                    }
                }
            }

            // ── Global fallback: scan every loader in the graph ──────
            //    The loader list is the same for every incrementer in the graph,
            //    so it is built once per graph version and shared.
            const gc = graphCache();
            if (!gc.candidates) {
                const all = [];
                for (const n of (G()._nodes || G().nodes || [])) {
                    if (!isInputLoader(n)) continue;
                    const src = getSourceFromNode(n);
                    if (!src) continue;
                    const blob = ((n.comfyClass || "") + " " + (n.title || "")).toLowerCase();
                    const moving = src.kind === "video" || src.kind === "sequence" || src.kind === "folder";
                    const isVideo = moving || /video|vhs/.test(blob);
                    const isImage = (src.kind === "image" || src.kind === "folder"
                        || (/image/.test(blob) && src.kind !== "video"));
                    // "many LoadVideo nodes" disambiguator: a loader whose output is
                    // actually wired into the pipeline is far more likely to be the
                    // one the user cares about than an orphaned/spare loader.
                    const isConnected = (n.outputs || []).some(o => o && o.links && o.links.length);
                    all.push({ node: n, filename: src.filename, isVideo, isImage, isConnected });
                }
                gc.candidates = all;
            }
            const candidates = gc.candidates.filter(c => c.node !== node);
            // BUG-FIX (Apr 2026): when explicit choice is set, drop
            // mismatched candidates entirely so we don't return the
            // wrong-type filename just because nothing of the right
            // type happens to be in the graph yet.
            let pool = candidates;
            if (choice === "video") pool = candidates.filter(c => c.isVideo);
            else if (choice === "image") pool = candidates.filter(c => c.isImage);
            else if (choice === "exr") {
                // Exr plate: exr/dpx/cin files, or any sequence (isVideo).
                pool = candidates.filter(c => isExrFile(c.filename) || c.isVideo);
            }
            if (pool.length === 0) return null;

            const score = (c) => {
                // Connected (wired-in) loaders outrank orphans by a wide margin
                // so "many LoadVideo nodes" resolves to the one in the pipeline.
                let s = c.isConnected ? 10 : 0;
                if (choice === "video")      s += c.isVideo ? 2 : (c.isImage ? 0 : 1);
                else if (choice === "image") s += c.isImage ? 2 : (c.isVideo ? 0 : 1);
                else                         s += c.isVideo ? 2 : (c.isImage ? 1 : 0); // auto
                return s;
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
            node._fiSyncedVersion = _fiVersion;   // see _fiVersion: later serializes skip until a change
            node._fiSyncedAt = performance.now();
            // source_path (a wired STRING path, e.g. from OCIORead / LoadEXRMEC
            // / a VFX plate loader) takes precedence over graph auto-detection.
            // Show the detected stem so the user can SEE the node found the EXR
            // without connecting a trigger — Python resolves the stem from
            // source_path at execution time regardless.
            const spInput = node.inputs?.find(i => i.name === "source_path");
            if (spInput && spInput.link != null) {
                const origin = getSourceNodeFromInput("source_path");
                const raw = origin ? getFilenameFromNode(origin) : null;
                const stem = stemFromPath(raw);
                if (stem) {
                    setStatus(`🔗 ${stem} (path mode)`);
                } else {
                    setStatus("🔗 path mode (stem resolved at run)");
                }
                return;   // setStatus redraws only when the text changed
            }
            const result = extractFilename();
            const sfWidget = node.widgets?.find(w => w.name === "source_filename");
            if (result && result.filename) {
                // source_filename = stem only; extension → source_extension
                const { stem, ext } = writeSourceWidgets(result.filename);
                const fmt      = getNameFormat();
                const preview  = formatSourceName(stem, fmt);
                const sfx      = (node.widgets?.find(w => w.name === "suffix")?.value || "").trim();
                const smode    = (node.widgets?.find(w => w.name === "suffix_mode")?.value || "filename").toString();
                let sfxLabel = "";
                if (sfx) {
                    const asFolder = sfx.replace(/^[_\-. ]+/, "");
                    if (smode === "subfolder")   sfxLabel = ` → /${asFolder}`;
                    else if (smode === "folder") sfxLabel = ` ⊞ ${sfx}`;
                    else                          sfxLabel = ` ${sfx}`;
                }
                const extLabel = ext ? ` ${ext}` : "";
                const tag = result.mode === "global" ? "\uD83C\uDF10"  // globe for global scan
                          : result.mode === "input"  ? "\uD83D\uDD0C"  // plug for non-trigger input
                          : result.mode === "custom" ? "\u270D\uFE0F"  // hand-writing for custom
                                                     : "\uD83D\uDCC4"; // page for trigger
                setStatus(`${tag} ${preview}${extLabel}${sfxLabel}`);   // redraws only on change
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

        // A sync that must not trust the per-version cache: an edit made in code
        // (no graphChanged) leaves _fiVersion where it was. Dropping this graph's
        // shared cache makes the next read rebuild it from the live widgets; the
        // rebuilt cache stays valid for the other incrementers of this version.
        function syncFresh() {
            _fiCaches.delete(G());
            normalizeSourceFilename();
            syncSourceFilename();
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
                if (node._fiWriting) return;      // our own write: already in sync
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
            _fiBump();
            setTimeout(syncSourceFilename, 150);
        };

        // Re-sync when source_choice widget changes
        const choiceWidget = node.widgets?.find(w => w.name === "source_choice");
        if (choiceWidget) {
            const origCb = choiceWidget.callback;
            choiceWidget.callback = function (v) {
                origCb?.apply(this, arguments);
                _fiBump();
                setTimeout(syncSourceFilename, 50);
            };
        }

        // Re-sync when name_format widget changes (live preview)
        const fmtWidget = node.widgets?.find(w => w.name === "name_format");
        if (fmtWidget) {
            const origCb = fmtWidget.callback;
            fmtWidget.callback = function (v) {
                origCb?.apply(this, arguments);
                if (!node._fiWriting) setTimeout(syncSourceFilename, 0);
            };
        }

        // Re-sync when custom_name widget changes (so the status preview
        // updates live in 'custom' mode).
        const customWidget = node.widgets?.find(w => w.name === "custom_name");
        if (customWidget) {
            const origCb = customWidget.callback;
            customWidget.callback = function (v) {
                origCb?.apply(this, arguments);
                if (!node._fiWriting) setTimeout(syncSourceFilename, 0);
            };
        }

        // Re-sync when suffix / suffix_mode change so the on-node status preview
        // reflects where the tag lands (filename vs subfolder vs folder) live.
        for (const wName of ["suffix", "suffix_mode"]) {
            const w = node.widgets?.find(x => x.name === wName);
            if (!w) continue;
            const origCb = w.callback;
            w.callback = function (v) {
                origCb?.apply(this, arguments);
                if (!node._fiWriting) setTimeout(syncSourceFilename, 0);
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
            // Runs on EVERY serialize (the change tracker's included), so it must be
            // cheap: re-sync only if the graph changed since this node last synced.
            // A queue right after an edit still sends a fresh value.
            // The 10 s cap covers an edit made in code without a graphChanged event,
            // on front-ends that lack the beforeQueued hook below.
            if (node._fiSyncedVersion !== _fiVersion) {
                normalizeSourceFilename();
                syncSourceFilename();
            } else if (performance.now() - (node._fiSyncedAt || 0) > 10000) {
                syncFresh();
            }
            origOnSerialize?.apply(this, arguments);
        };

        // Fresh name for the prompt itself: the front-end calls every widget's
        // beforeQueued right before graphToPrompt (core uses it for seeds), so a
        // queue never sends a stale source, without paying on every serialize.
        const sfQueueWidget = node.widgets?.find(w => w.name === "source_filename");
        if (sfQueueWidget) {
            const prevBeforeQueued = sfQueueWidget.beforeQueued;
            sfQueueWidget.beforeQueued = function (...args) {
                try { syncFresh(); } catch (_) { /* never block a queue */ }
                return prevBeforeQueued?.apply(this, args);
            };
        }

        // Initial sync + periodic retry (graph may not be fully loaded yet)
        hideExtensionWidget();
        normalizeSourceFilename();
        setTimeout(() => { hideExtensionWidget(); normalizeSourceFilename(); syncSourceFilename(); }, 500);
        setTimeout(() => { hideExtensionWidget(); normalizeSourceFilename(); syncSourceFilename(); }, 2000);

        // Changes elsewhere in the graph (another loader's file, a new
        // Set/Get pair) arrive through the page-wide `graphChanged` listener
        // above - no per-node timer.
        if (node._fiPollTimer) { clearInterval(node._fiPollTimer); node._fiPollTimer = null; }
        _fiLive.add(node);
        const _origRemoved = node.onRemoved;
        node.onRemoved = function () {
            _fiLive.delete(node);
            return _origRemoved?.apply(this, arguments);
        };
    },
});
