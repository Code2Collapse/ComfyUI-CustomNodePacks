/**
 * Save Video (C2C) — L9.40 (owner 2026-10-23, docs/save-video-v2.md):
 *
 * - Format: format = custom shows container -> codec -> profile -> bit depth -> chroma, each list filtered by the
 *   ones before it from the server's matrix (/c2c/save_video/matrix: what THIS machine can write, with the reason for
 *   what it cannot). A workflow saved with an old single-list label is converted to the same five choices when it
 *   loads (the label alone would still encode the same file). Colour tags show where the codec takes them.
 * - Naming: under the naming control the node shows the path the next write will use (Folder Version Incrementer
 *   in the graph, or its defaults; /c2c/save_video/next_path, a dry run), refreshed when a naming input changes.
 * - Player: Load Video's player on the WHOLE trimmed clip (the run's source proxy). Its In / Out handles are the
 *   export range (range_in / range_out). A/B against the saved file (wipe, difference, onion, flicker), zoom, pan,
 *   1:1 and a loupe (js/_c2c_ab_view.js).
 * - Preview first: a run shows what would be written and writes nothing; "Write file" queues again with the run's
 *   fingerprint, and only the encode runs (the graph above is cached).
 * - Handles: trim_frames / trim_side show only for a manual trim.
 *
 * Nothing polls: the matrix is fetched once, the next path on naming changes, the player works only while playing.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { measureRootContent, installResizeFloor } from "./c2c_ui/index.js";
import { setHidden, vueSyncNodeWidgets } from "./_widget_visibility.js";
import {
    MARGIN,
    DEFAULT_ASPECT,
    buildDom,
    setLoading,
    renderChips,
    attachPauseObservers,
    reservedHeight,
} from "./c2c_video_player_ui.js";
import { attachPlayer, holdPlayerWidth, playerButton } from "./c2c_video_player.js";
import { attachABView, WIPE_SHAPES } from "./_c2c_ab_view.js";

const NODE = "SaveVideoC2C";
const SETTING_NAMING = "c2c.saveVideo.naming";
const CUSTOM = "custom (container / codec)";
const FIVE = ["container", "codec", "profile", "bit_depth", "chroma"];
// L9.54 (owner 2026-10-25): Format -> Subformat -> only that subformat's options; less-used ones under Advanced
const SUB_OPTS = ["quality", "encoder_preset", "profile", "bit_depth", "chroma", "gop", "bitrate_kbps", "channels",
                  "png_compression"];
// what a subformat lists shows (owner's per-format lists: EXR channels, H.264 preset ...); keyframes, bitrate and the
// colour tags are the "advanced" ones (owner: Advanced = timecode, range, colour metadata, encoder-specific)
const ADVANCED = new Set(["gop", "bitrate_kbps", "color_primaries", "color_transfer", "color_range"]);
const ENCODER_EXTRAS = new Set(["libx264", "libx265", "h264_nvenc", "hevc_nvenc", "av1_nvenc", "libsvtav1",
                                "libvpx-vp9"]);
const COLOUR = ["color_primaries", "color_transfer", "color_range"];
const ALWAYS_HIDDEN = ["naming_resolved", "write_token", "range_in", "range_out"];
const NAMING_INPUTS = new Set(["naming", "filename_prefix", "subfolder", "save_output", "format", "subformat", ...FIVE]);
const FOLDER_NAMES = new Set(["Folder Version Incrementer", "Folder Version"]);

// the old single list: formats without a quality setting (legacy labels only; the matrix says it for custom)
const NO_QUALITY_FORMATS = new Set([
    "MOV DNxHR HQ", "MOV DNxHR HQX", "MOV DNxHR 444", "MKV FFV1 (lossless)",
    "PNG sequence 16-bit", "EXR sequence (half)", "EXR sequence (float)",
]);

let MATRIX = null;
let matrixPromise = null;
function loadMatrix() {
    matrixPromise ??= api.fetchApi("/c2c/save_video/matrix")
        .then((r) => r.json())
        .then((m) => { if (m && !m.error) MATRIX = m; return MATRIX; })
        .catch(() => null);
    return matrixPromise;
}

const W = (node, name) => node.widgets?.find((w) => w.name === name);
const val = (node, name, d) => W(node, name)?.value ?? d;
const baseLabel = (label) => String(label || "").replace(/\s+\[unavailable\]$/, "").trim();

function viewUrl(meta) {
    if (!meta?.filename) return "";
    return api.apiURL(
        `/view?filename=${encodeURIComponent(meta.filename)}`
        + `&type=${encodeURIComponent(meta.type || "output")}`
        + `&subfolder=${encodeURIComponent(meta.subfolder || "")}`,
    );
}

function resolveNaming(node) {
    const resolvedW = W(node, "naming_resolved");
    if (!resolvedW) return;
    const naming = val(node, "naming", "default");
    if (naming === "default") {
        try {
            resolvedW.value = app.ui?.settings?.getSettingValue?.(SETTING_NAMING, "Folder Version Incrementer")
                ?? "Folder Version Incrementer";
        } catch {
            resolvedW.value = "Folder Version Incrementer";
        }
    } else {
        resolvedW.value = naming;
    }
}

function effectiveNaming(node) {
    const n = val(node, "naming", "default");
    return n === "default" ? (val(node, "naming_resolved", "") || "Folder Version Incrementer") : n;
}

// ── format: the five controls ─────────────────────────────────────────────────────────────────────────────────
function setOptions(w, values) {
    if (!w || !Array.isArray(values) || !values.length) return;
    w.options = w.options || {};
    w.options.values = values;
}

function snap(w, values, notes, what) {
    if (!w || !values?.length) return;
    if (!values.includes(w.value)) {
        notes.push(`${what} ${w.value} is not offered here: ${values[0]}`);
        w.value = values[0];
    }
}

/** Filter each control by the ones before it; returns {notes, codec info}. Writes values directly (no callbacks). */
function cascade(node) {
    const notes = [];
    if (!MATRIX) return { notes, info: null };
    const cW = W(node, "container"), kW = W(node, "codec"), pW = W(node, "profile");
    const bW = W(node, "bit_depth"), hW = W(node, "chroma");
    if (!cW || !kW || !pW || !bW || !hW) return { notes, info: null };
    setOptions(cW, Object.keys(MATRIX.containers));
    const codecs = MATRIX.containers[cW.value] || [];
    setOptions(kW, codecs);
    snap(kW, codecs, notes, "codec");
    const info = MATRIX.codecs[kW.value];
    if (!info) return { notes, info: null };

    const profiles = Object.keys(info.profiles);
    setOptions(pW, profiles);
    snap(pW, profiles, notes, "profile");
    const modes = info.profiles[pW.value] || [];
    const bits = [...new Set(modes.map((m) => m.bits))];
    setOptions(bW, bits);
    snap(bW, bits, notes, "bit depth");
    const chromas = modes.filter((m) => m.bits === bW.value).map((m) => m.chroma);
    setOptions(hW, chromas);
    snap(hW, chromas, notes, "chroma");
    if (info.reason) notes.push(`${kW.value} is not available here: ${info.reason}`);
    const mode = modes.find((m) => m.bits === bW.value && m.chroma === hW.value);
    if (mode?.reason) notes.push(`${kW.value} ${pW.value} ${bW.value}-bit ${hW.value}: ${mode.reason}`);
    return { notes, info };
}

/** A workflow saved with an old single-list label: the same file as the five controls (format -> custom). */
function migrateLegacy(node) {
    const fW = W(node, "format");
    if (!MATRIX?.legacy || !fW) return null;
    const label = baseLabel(fW.value);
    const five = MATRIX.legacy[label];
    if (!five || !FIVE.every((n) => W(node, n))) return null;
    FIVE.forEach((n, i) => { W(node, n).value = five[i]; });
    fW.value = CUSTOM;
    return label;
}

// ── L9.54 subformats ──────────────────────────────────────────────────────────────────────────────────────────
const hasSubformats = () => !!(MATRIX?.subformats && MATRIX.families);

function subformatsOf(family) {
    return Object.entries(MATRIX?.subformats || {}).filter(([, d]) => d.family === family).map(([k]) => k);
}

/** The subformat the five controls belong to (exact match, else same container + codec), or null (= Custom). */
function deriveSubformat(node) {
    const five = FIVE.map((n) => String(val(node, n)));
    const subs = Object.entries(MATRIX?.subformats || {});
    const exact = subs.find(([, d]) => d.five.every((v, i) => String(v) === five[i]));
    if (exact) return exact[0];
    const same = subs.find(([, d]) => d.five[0] === five[0] && d.five[1] === five[1]);
    return same ? same[0] : null;
}

function memory(node) {
    node.properties = node.properties || {};
    node.properties.c2c_subformat_memory = node.properties.c2c_subformat_memory || {};
    return node.properties.c2c_subformat_memory;
}

/** Remember this subformat's current values (owner: "remember the last valid settings for each subformat"). */
function rememberSubformat(node) {
    const sub = val(node, "subformat");
    if (!sub || !MATRIX?.subformats?.[sub]) return;
    const m = {};
    for (const n of [...FIVE, ...SUB_OPTS]) if (W(node, n)) m[n] = W(node, n).value;
    memory(node)[sub] = m;
}

/** Switch to a subformat: its remembered values, else its defaults, written straight into the controls. */
function applySubformat(node, sub) {
    const d = MATRIX?.subformats?.[sub];
    if (!d) return;
    const mem = memory(node)[sub];
    FIVE.forEach((n, i) => { const w = W(node, n); if (w) w.value = d.five[i]; });
    if (mem) {
        for (const [n, v] of Object.entries(mem)) {
            if (n === "container" || n === "codec") continue;   // fixed by the subformat
            const w = W(node, n);
            if (w) w.value = v;
        }
    }
}

/** Keep family -> subformat lists consistent; returns the active subformat label or null (Custom). */
function syncFamily(node) {
    const fW = W(node, "format_family"), sW = W(node, "subformat");
    if (!fW || !sW || !hasSubformats()) return null;
    if (fW.value === "Custom") {
        setOptions(sW, [MATRIX.from_controls]);
        sW.value = MATRIX.from_controls;
        return null;
    }
    const subs = subformatsOf(fW.value);
    setOptions(sW, subs.length ? subs : [MATRIX.from_controls]);
    if (!subs.includes(sW.value)) {
        sW.value = subs[0];
        applySubformat(node, subs[0]);
    }
    return sW.value;
}

/** A saved workflow (no subformat yet): show it on the subformat its five controls belong to - same encode. */
function adoptSubformat(node) {
    const fW = W(node, "format_family"), sW = W(node, "subformat");
    if (!fW || !sW || !hasSubformats() || sW.value !== MATRIX.from_controls || val(node, "format") !== CUSTOM) return;
    const sub = deriveSubformat(node);
    if (sub) {
        fW.value = MATRIX.subformats[sub].family;
        setOptions(sW, subformatsOf(fW.value));
        sW.value = sub;
    } else {
        fW.value = "Custom";
    }
}

/** Can the current profile / depth / chroma carry alpha? (ProRes: only 4444 / XQ - the matrix says per mode.) */
function modeAlpha(node, info) {
    const mode = (info?.profiles?.[val(node, "profile")] || [])
        .find((m) => m.bits === String(val(node, "bit_depth")) && m.chroma === val(node, "chroma"));
    return mode?.alpha ?? !!info?.alpha;
}

/** ProRes: every profile is its own subformat, so a profile picked inside one moves to that subformat.
 *  Returns true when the subformat changed. */
function followProfile(node) {
    const sW = W(node, "subformat"), cur = MATRIX?.subformats?.[sW?.value];
    const p = String(val(node, "profile"));
    if (!cur || String(cur.five[2]) === p) return false;
    const other = subformatsOf(cur.family).find((k) => {
        const d = MATRIX.subformats[k];
        return d.five[0] === cur.five[0] && d.five[1] === cur.five[1] && String(d.five[2]) === p;
    });
    if (!other) return false;
    sW.value = other;
    return true;
}

function setLabel(w, text) {
    if (!w) return;
    if (w.__c2cBaseLabel === undefined) w.__c2cBaseLabel = (w.label && w.label !== w.name) ? w.label : null;
    w.label = text ?? w.__c2cBaseLabel ?? w.name;
}

function applyVisibility(node, info) {
    const custom = val(node, "format") === CUSTOM;
    const subMode = custom && hasSubformats() && !!W(node, "subformat");
    const adv = !!node.properties?.c2c_save_advanced;
    const sub = subMode ? val(node, "subformat") : null;
    const sd = sub ? MATRIX.subformats?.[sub] : null;
    const enc = info?.enc;
    if (subMode) {
        setHidden(W(node, "format"), true);                  // the old single list: its entries are the subformats now
        setHidden(W(node, "format_family"), false);
        setLabel(W(node, "format_family"), "format");        // the owner's "Format -> Subformat"
        setHidden(W(node, "subformat"), !sd);
        for (const n of ["container", "codec"]) setHidden(W(node, n), !!sd);
        const shown = (n) => {
            if (n === "channels") return (!sd || sd.options.includes(n)) && modeAlpha(node, info);
            if (sd) return sd.options.includes(n);
            if (FIVE.includes(n)) return true;               // Custom: the cascade
            if (n === "quality") return !!info?.quality;
            if (n === "png_compression") return enc === "png" || enc === "oiio_png";
            return ENCODER_EXTRAS.has(enc);                  // preset / gop / bitrate
        };
        for (const n of SUB_OPTS) setHidden(W(node, n), !shown(n) || (ADVANCED.has(n) && !adv));
        for (const n of SUB_OPTS) setLabel(W(node, n), sd ? MATRIX.subformats[sub].labels?.[n] : null);
        const colour = info?.colour ?? "yuv";
        setHidden(W(node, "color_primaries"), colour === "none" || !adv);
        setHidden(W(node, "color_transfer"), colour === "none" || !adv);
        setHidden(W(node, "color_range"), colour !== "yuv" || !adv);
    } else {
        for (const n of ["format_family", "subformat", "encoder_preset", "gop", "bitrate_kbps", "channels",
                         "png_compression"]) setHidden(W(node, n), true);
        for (const n of FIVE) setHidden(W(node, n), !custom);
        const quality = custom ? !!info?.quality : !NO_QUALITY_FORMATS.has(baseLabel(val(node, "format")));
        setHidden(W(node, "quality"), !quality);
        const colour = custom ? (info?.colour ?? "yuv") : "yuv";
        setHidden(W(node, "color_primaries"), colour === "none");
        setHidden(W(node, "color_transfer"), colour === "none");
        setHidden(W(node, "color_range"), colour !== "yuv");
    }
    // the stream-save options matter only with a latent wired (images arrive decoded otherwise)
    const streamed = node.inputs?.find((i) => i.name === "latent")?.link != null;
    setHidden(W(node, "tile_mode"), !streamed);
    setHidden(W(node, "tile_size"), !streamed || val(node, "tile_mode") !== "manual");
    const manual = val(node, "handles_trim") === "manual";
    setHidden(W(node, "trim_frames"), !manual);
    setHidden(W(node, "trim_side"), !manual);
    for (const n of ALWAYS_HIDDEN) setHidden(W(node, n), true);
    vueSyncNodeWidgets(node);
    const sz = node.computeSize?.();
    if (sz) node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
    node.setDirtyCanvas?.(true, true);
}

// ── the widget ────────────────────────────────────────────────────────────────────────────────────────────────
function mountSaveWidget(node) {
    const cfg = { hasPreview: true, emptyTitle: "Save Video (C2C)", emptyHint: "Queue to write and preview the file." };
    const ui = buildDom(cfg);
    let aspect = DEFAULT_ASPECT;
    let last = null;
    let ab = null;
    let shown = false;          // L9.52: before a run there is nothing to show - no stage, no player, no A/B tools

    // the extra rows live in ui.summary, which reservedHeight() counts
    const summary = document.createElement("div");
    summary.className = "c2c-vloader__summary";
    const pathLine = document.createElement("div");
    pathLine.className = "c2c-vloader__summary-lines";
    pathLine.dataset.c2cSave = "next-path";
    const formatLine = document.createElement("div");
    formatLine.className = "c2c-vloader__summary-lines";
    formatLine.dataset.c2cSave = "format-notes";
    const tools = document.createElement("div");
    tools.className = "c2c-vloader__summary-chips";
    tools.dataset.c2cSave = "tools";
    const advRow = document.createElement("div");
    advRow.className = "c2c-vloader__summary-chips";
    const bAdv = playerButton("Advanced ▸", "Show the less-used options: encoder preset, keyframes, bitrate, alpha, "
        + "PNG compression, colour tags", () => {
        node.properties = node.properties || {};
        node.properties.c2c_save_advanced = !node.properties.c2c_save_advanced;
        refreshFormat();
    });
    bAdv.dataset.c2cSave = "advanced";
    advRow.append(bAdv);
    summary.append(tools, advRow, pathLine, formatLine);
    ui.root.insertBefore(summary, ui.status.el);
    ui.summary = summary;

    const nodeWidth = () => node.size?.[0] || 360;
    const getReserved = (w) => reservedHeight(w || nodeWidth(), shown, aspect, ui);
    const fitNode = () => requestAnimationFrame(() => {
        const need = node.computeSize?.();
        if (!need || !node.setSize) return;
        node.setSize([node.size[0], need[1]]);
        node.setDirtyCanvas?.(true, true);
    });

    const widgetRef = node.addDOMWidget("c2c_preview", "div", ui.root, {
        serialize: false,
        margin: MARGIN,
        getMinHeight: () => getReserved(nodeWidth()),
    });
    widgetRef.computeSize = (w) => [w, getReserved(w)];
    const uninstallFloor = installResizeFloor(ui.root, measureRootContent);
    const cleanupVideo = attachPauseObservers(node, ui.video);

    // player on the whole trimmed clip; its selection is the export range
    ui.track.hidden = true;
    ui.labelLeft.parentElement.hidden = true;
    const player = attachPlayer(ui.timeline, ui.video, {
        fullscreenTarget: ui.root,
        verb: "write",
        inTitle: "First frame to write (export range In)",
        outTitle: "Last frame to write (export range Out)",
        onSelect: (inF, outF) => {
            const total = last?.frames_total || 0;
            const whole = inF <= 0 && total && outF >= total - 1;
            const a = W(node, "range_in"), b = W(node, "range_out");
            if (a) a.value = whole ? 0 : inF;
            if (b) b.value = whole ? 0 : outF;
            ui.status.setText(whole ? "Export range: the whole clip." :
                `Export range: frames ${inF}-${outF} (${outF - inF + 1}). Queue to write it.`, "default");
            node.setDirtyCanvas?.(true, true);
        },
    });
    ab = attachABView(ui.viewport, ui.video, { labels: ["source", "saved"] });
    holdPlayerWidth(node, player);                      // the transport on ONE line (L9.52)

    // A/B + view tools
    const modeBtns = {};
    const setMode = (m) => {
        ab.setMode(m);
        for (const [k, b] of Object.entries(modeBtns)) b.setAttribute("aria-pressed", String(k === m));
    };
    const addMode = (m, label, title) => { modeBtns[m] = playerButton(label, title, () => setMode(m)); return modeBtns[m]; };
    let shapeIdx = 0;
    const bShape = playerButton("|", "Wipe direction: vertical, horizontal, diagonal", () => {
        shapeIdx = (shapeIdx + 1) % WIPE_SHAPES.length;
        ab.setShape(WIPE_SHAPES[shapeIdx]);
        bShape.textContent = ["|", "—", "/"][shapeIdx];
    });
    let gain = 1;
    const bGain = playerButton("×1", "Difference gain (click: x2, shift-click: /2)", () => {});
    bGain.addEventListener("click", (e) => {
        gain = e.shiftKey ? Math.max(1, gain / 2) : Math.min(64, gain * 2);
        ab.setGain(gain);
        bGain.textContent = `×${gain}`;
    }, { capture: true });
    let loupeOn = false;
    const bLoupe = playerButton("loupe", "Magnifier with pixel values under the pointer", () => {
        loupeOn = !loupeOn;
        ab.setLoupe(loupeOn);
        bLoupe.setAttribute("aria-pressed", String(loupeOn));
    });
    const bWrite = playerButton("Write file", "Write exactly what the preview shows (only the encode runs again)",
        () => writeNow());
    bWrite.dataset.c2cSave = "write";
    bWrite.hidden = true;
    tools.append(
        addMode("A", "source", "The frames that came in (handles trimmed)"),
        addMode("B", "saved", "The written file"),
        addMode("wipe", "A|B", "Wipe between source and saved (drag the line)"), bShape,
        addMode("difference", "diff", "|source - saved|: codec and colour damage shows bright"), bGain,
        addMode("onion", "onion", "Saved over source at 50 %"),
        addMode("flicker", "flicker", "Source and saved in turn"),
        playerButton("fit", "Fit (or double-click the picture)", () => ab.fit()),
        playerButton("1:1", "One file pixel per screen pixel", () => ab.oneToOne()),
        bLoupe, bWrite,
    );
    setMode("A");

    const setShown = (on) => {
        shown = on;
        if (ui.previewWrap) ui.previewWrap.hidden = !on;
        ui.timeline.hidden = !on;
        tools.hidden = !on;
    };
    setShown(false);
    ui.status.setText("Run the graph to preview what will be written.", "default");

    async function writeNow() {
        const fp = last?.fingerprint;
        const tw = W(node, "write_token");
        if (!fp || !tw) return;
        tw.value = fp;
        bWrite.disabled = true;
        try {
            await app.queuePrompt(0, 1);
            ui.status.setText("Writing the file...", "default");
        } finally {
            tw.value = "";               // the queued prompt has the token; the next run previews again
        }
    }

    // ── next path (dry run) ──
    let pathTimer = 0;
    let pathSeq = 0;
    async function refreshPath() {
        const seq = ++pathSeq;
        const naming = effectiveNaming(node);
        const body = {
            naming: val(node, "naming", "default"),
            format: val(node, "format"), filename_prefix: val(node, "filename_prefix", ""),
            subfolder: val(node, "subfolder", ""), save_output: !!val(node, "save_output", true),
            unique_id: String(node.id),
        };
        for (const n of FIVE) body[n] = val(node, n);
        if (FOLDER_NAMES.has(naming) && !String(body.subfolder || "").trim()) {
            try { body.prompt = (await app.graphToPrompt())?.output || {}; } catch { body.prompt = {}; }
        }
        try {
            const r = await api.fetchApi("/c2c/save_video/next_path", { method: "POST", body: JSON.stringify(body) });
            const d = await r.json();
            if (seq !== pathSeq) return;                          // a newer request is on its way
            pathLine.textContent = d.error ? `Next path: ${d.error}`
                : `Next: ${d.path}${d.origin ? ` (${d.origin})` : ""}`;
        } catch {
            if (seq === pathSeq) pathLine.textContent = "Next path: the server did not answer.";
        }
        fitNode();
    }
    const schedulePath = () => { clearTimeout(pathTimer); pathTimer = setTimeout(refreshPath, 450); };

    // ── format notes + visibility ──
    function refreshFormat() {
        if (hasSubformats() && val(node, "format") === CUSTOM) syncFamily(node);
        const { notes, info } = cascade(node);
        applyVisibility(node, info);
        const adv = !!node.properties?.c2c_save_advanced;
        bAdv.textContent = adv ? "Advanced ▾" : "Advanced ▸";
        bAdv.setAttribute("aria-pressed", String(adv));
        bAdv.hidden = !(hasSubformats() && val(node, "format") === CUSTOM);
        const migrated = node._c2cSaveMigrated;
        const preset = node._c2cSavePreset;
        const lines = [];
        if (migrated) lines.push(`Converted from the old format "${migrated}" (same file).`);
        if (preset) lines.push(`Preset "${preset}": ${FIVE.map((n) => val(node, n)).join(" / ")}.`);
        lines.push(...notes);
        formatLine.textContent = lines.join(" · ");
        formatLine.hidden = !lines.length;
        fitNode();
    }

    // ── a run's result ──
    const showResult = (data) => {
        if (!data) return;
        last = data;
        setShown(true);
        aspect = (data.width && data.height) ? data.width / data.height : DEFAULT_ASPECT;
        if (ui.viewport) ui.viewport.style.aspectRatio = String(aspect);
        renderChips(ui.info, data);
        for (const t of [data.profile ? `profile ${data.profile}` : "", data.pix_fmt || "",
                         data.version?.version_string ? `version ${data.version.version_string}` : ""]) {
            if (!t) continue;
            const chip = document.createElement("span");
            chip.className = "c2c-vloader__chip";
            chip.textContent = t;
            ui.info.appendChild(chip);
        }
        const preview = data.mode === "preview";
        const aMeta = data.source || data.preview;
        const total = data.frames_total || data.frames || 1;
        const url = viewUrl(aMeta);
        if (url) {
            setLoading(ui, true, aspect);
            ui.viewport?.querySelector(".c2c-ui-empty")?.remove();
            ui.video.src = url;
            ui.video.onloadeddata = () => setLoading(ui, false, aspect);
            ui.video.onerror = () => { setLoading(ui, false, aspect); ui.status.setText("Could not play the preview.", "danger"); };
        }
        ab.setFpsA(data.fps || 24);
        player.setDomain({ frames: total, fps: data.fps || 24, step: 1, previewFrames: total, previewFps: data.fps || 24 });
        const rIn = Number(data.range_in || 0), rOut = Number(data.range_out || 0) || total - 1;
        player.setSelection({ in: rIn, out: rOut, count: rOut - rIn + 1, nth: 1 });
        const savedUrl = !preview && data.source ? viewUrl(data.preview) : "";
        ab.setB(savedUrl, { fps: data.fps || 24, frames: data.frames || 0, offset: rIn });
        for (const m of ["B", "wipe", "difference", "onion", "flicker"]) modeBtns[m].disabled = !savedUrl;
        if (!savedUrl) setMode("A");
        bShape.disabled = bGain.disabled = !savedUrl;
        bWrite.hidden = !preview;
        bWrite.disabled = !preview || !data.fingerprint;
        const parts = [];
        if (data.message) parts.push(data.message);
        if (data.handles?.head || data.handles?.tail) parts.push(`Handles trimmed: ${data.handles.from}.`);
        else if (data.handles?.from && !/^trim off|nothing trimmed/.test(data.handles.from)) parts.push(data.handles.from);
        if (data.range) parts.push(`Wrote ${data.range}.`);
        if (data.colour) {                                        // L9.53: what Save did with the colour
            const c = data.colour;
            parts.push(c.transform_text ? `Colour: ${c.reverses_load ? "reversed Load Video - " : ""}${c.transform_text}.`
                : `Colour: ${c.out} (not converted).`);
        }
        if (data.version?.origin) parts.push(`Version ${data.version.version_string} from the ${data.version.origin}.`);
        if (!preview && !data.source) parts.push("A/B: this run kept no source copy (streamed save) - the saved file only.");
        const warns = Array.isArray(data.warnings) ? data.warnings : [];
        ui.status.setText([...parts, ...warns].join(" "), warns.length ? "warn" : "default");
        schedulePath();                                           // the version advanced
        fitNode();
    };
    node._c2cSaveShowResult = showResult;

    const origRemoved = node.onRemoved;
    node.onRemoved = function (...args) {
        clearTimeout(pathTimer);
        player.destroy();
        ab.destroy();
        cleanupVideo();
        uninstallFloor();
        return origRemoved?.apply(this, args);
    };
    return { refreshFormat, schedulePath };
}

// L9.54 (owner: "organized like VHS Video Combine"): Format -> Subformat -> its options show FIRST. The new inputs are
// appended last in Python (saved workflows keep their positions), so only the display order changes: widgets_values
// is written back in the input order (onSerialize) and restored by name (onConfigure) - frontend 1.53 restores by
// position and also writes widgets_values_named, so workflows load the same either way.
const FROM_LOAD = "auto (from Load)";                // L9.53, nodes/c2c_video/colour_cm.py FROM_LOAD

const DISPLAY_FIRST = ["format", "format_family", "subformat", "container", "codec", "profile", "bit_depth", "chroma",
                       "quality", "encoder_preset", "gop", "bitrate_kbps", "channels", "png_compression",
                       "color_primaries", "color_transfer", "color_range"];

function orderForDisplay(node) {
    if (!node.widgets?.length || node._c2cCanon) return;
    node._c2cCanon = node.widgets.filter((w) => w.serialize !== false).map((w) => w.name);
    node._c2cDefaults = Object.fromEntries(node.widgets.map((w) => [w.name, w.value]));
    const first = DISPLAY_FIRST.map((n) => W(node, n)).filter(Boolean);
    const rest = node.widgets.filter((w) => !first.includes(w));
    node.widgets.splice(0, node.widgets.length, ...first, ...rest);
}

/** The input order of the serialized widgets (widgets added after creation keep their place at the end). */
function canonicalWidgets(node) {
    const ser = node.widgets.filter((w) => w.serialize !== false);
    const byName = new Map(ser.map((w) => [w.name, w]));
    const canon = node._c2cCanon.map((n) => byName.get(n)).filter(Boolean);
    return [...canon, ...ser.filter((w) => !canon.includes(w))];
}

function restoreByName(node, info) {
    const vals = info?.widgets_values;
    if (!node._c2cCanon || !Array.isArray(vals)) return;
    const named = info.widgets_values_named;
    canonicalWidgets(node).forEach((w, i) => {
        if (named && Object.hasOwn(named, w.name)) w.value = named[w.name];
        else if (i < vals.length) w.value = vals[i];
        else if (Object.hasOwn(node._c2cDefaults, w.name)) w.value = node._c2cDefaults[w.name];  // older workflow
    });
}

function hook(node, name, fn) {
    const w = W(node, name);
    if (!w || w.__c2cSvHooked) return;
    w.__c2cSvHooked = true;
    const orig = w.callback;
    w.callback = function (...args) {
        const r = orig?.apply(this, args);
        if (!app.configuringGraph) { try { fn(); } catch (_e) { /* a hint never breaks the widget */ } }
        return r;
    };
}

if (!globalThis.__c2cSaveVideoExt) {
    globalThis.__c2cSaveVideoExt = true;
    app.registerExtension({
        name: "C2C.SaveVideo",
        settings: [{
            id: SETTING_NAMING,
            name: "Save Video naming mode",
            type: "combo",
            options: ["Folder Version Incrementer", "ComfyUI counter"],
            defaultValue: "Folder Version Incrementer",
            category: ["c2c", "Video", "Save Video"],
            tooltip: "What naming 'default' means on Save Video (C2C): the version folder from the Folder Version "
                + "Incrementer (its node in the graph, or its defaults), or prefix_00001 like Save Image.",
        }],
        async setup() { loadMatrix(); },
        async beforeRegisterNodeDef(nodeType, nodeData) {
            if (nodeData.name !== NODE) return;
            const onCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function (...args) {
                const r = onCreated?.apply(this, args);
                if (this._c2cSaveMounted) return r;
                this._c2cSaveMounted = true;
                const node = this;
                const api_ = mountSaveWidget(node);
                node._c2cSave = api_;
                orderForDisplay(node);
                // L9.52: a named format is a PRESET - picking one fills the five controls (format -> custom), so its
                // options always show (they used to vanish: owner "controls didn't appear")
                hook(node, "format", () => {
                    const from = migrateLegacy(node);
                    if (from) { node._c2cSavePreset = from; node._c2cSaveMigrated = null; }
                    api_.refreshFormat();
                });
                for (const n of [...FIVE, "handles_trim", "tile_mode"]) {
                    hook(node, n, () => {
                        if (FIVE.includes(n)) node._c2cSavePreset = null;
                        api_.refreshFormat();
                        if (n === "profile" && followProfile(node)) api_.refreshFormat();
                        rememberSubformat(node);
                    });
                }
                hook(node, "format_family", () => { syncFamily(node); api_.refreshFormat(); rememberSubformat(node); });
                hook(node, "subformat", () => {
                    applySubformat(node, val(node, "subformat"));
                    api_.refreshFormat();
                    rememberSubformat(node);
                });
                for (const n of SUB_OPTS) {
                    if (!FIVE.includes(n)) hook(node, n, () => rememberSubformat(node));
                }
                // a node made now (not loaded from a workflow) starts 420 px wide; the frontend sizes it to
                // computeSize() after onNodeCreated, so this waits a tick
                const fresh = !app.configuringGraph;
                for (const n of NAMING_INPUTS) hook(node, n, () => { resolveNaming(node); api_.schedulePath(); });
                for (const n of ["naming", "format"]) {
                    const w = W(node, n);
                    if (!w) continue;
                    const prev = w.beforeQueued;
                    w.beforeQueued = function (...a) { resolveNaming(node); return prev?.apply(this, a); };
                }
                setTimeout(() => {
                    if (fresh && node.size?.[0] < 420) node.setSize([420, node.size[1]]);
                    if (fresh) {                       // L9.53: a new node reverses what Load Video did
                        for (const n of ["colorspace_in", "colorspace_out"]) {
                            const w = W(node, n);
                            if (w && w.options?.values?.includes(FROM_LOAD)) w.value = FROM_LOAD;
                        }
                    }
                    resolveNaming(node);
                    applyVisibility(node, null);
                    loadMatrix().then(() => {
                        if (fresh && hasSubformats() && W(node, "subformat")?.value === MATRIX.from_controls) {
                            W(node, "format_family").value = "Video";
                            W(node, "subformat").value = "H.264 / MP4";
                            applySubformat(node, "H.264 / MP4");
                        }
                        api_.refreshFormat();
                        api_.schedulePath();
                    });
                }, 0);
                return r;
            };
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (...args) {
                restoreByName(this, args[0]);               // before anything reads the values
                const r = onConfigure?.apply(this, args);
                const node = this;
                loadMatrix().then(() => {
                    const from = migrateLegacy(node);
                    if (from) node._c2cSaveMigrated = from;
                    adoptSubformat(node);                    // L9.54: shown on its subformat - the same encode
                    node._c2cSave?.refreshFormat();
                    node._c2cSave?.schedulePath();
                });
                return r;
            };
            const onSerialize = nodeType.prototype.onSerialize;
            nodeType.prototype.onSerialize = function (o) {
                onSerialize?.apply(this, arguments);
                if (!this._c2cCanon || !Array.isArray(o?.widgets_values)) return;
                o.widgets_values = canonicalWidgets(this).map((w) => {
                    const v = w.value;
                    return typeof v === "object" && v ? JSON.parse(JSON.stringify(v)) : v ?? null;
                });
            };
            const onConn = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (...args) {
                const r = onConn?.apply(this, args);
                if (this._c2cSave && !app.configuringGraph) this._c2cSave.refreshFormat();
                return r;
            };
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                const r = onExecuted?.apply(this, arguments);
                this._c2cSaveShowResult?.(message?.c2c_save_video?.[0]);
                return r;
            };
        },
    });
}
