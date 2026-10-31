/**
 * Load Video (C2C) as the unified loader (L9.89, owner A21 Rev 3 Part 3.4 / 3.12): the upload bar (files, a folder,
 * drag-and-drop, progress, explicit overwrite, refresh), the source chips (kind, pattern, frame numbers, missing
 * frames, EXR layers - a layer chip picks it), and the candidate list when a folder holds more than one sequence.
 *
 * Core's upload takes video/webm and video/mp4 only and EXR / DPX have no browser MIME type, so the bar posts to the
 * pack's own route (POST /c2c/media/upload, nodes/c2c_video/media_upload.py) with the formats this ComfyUI can read
 * (GET /c2c/media/formats). Nothing here holds state the workflow needs: the result is the `video` widget's value.
 */
import { api } from "../../scripts/api.js";
import { button } from "./c2c_ui/index.js";

const STYLE_ID = "c2c-loader-source-v1";

function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
.c2c-vloader__source { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
.c2c-vloader__source .c2c-ui-btn { padding: 2px 8px; font-size: 11px; min-height: 22px; }
.c2c-vloader__over { display: inline-flex; align-items: center; gap: 4px; font-size: 10px; color: var(--cu-ink-soft);
  cursor: pointer; user-select: none; }
.c2c-vloader__progress { flex: 1 1 100%; height: 4px; border-radius: 2px; background: var(--cu-sunken); overflow: hidden; }
.c2c-vloader__progress > div { height: 100%; width: 0%; background: var(--cu-accent, #6aa0ff); transition: width .1s linear; }
.c2c-vloader__chip--warn { border-color: var(--cu-danger, #d05050); color: var(--cu-danger, #d05050); }
.c2c-vloader__chip--pick { cursor: pointer; }
.c2c-vloader__chip--pick:hover { border-color: var(--cu-accent, #6aa0ff); }
.c2c-vloader__chip--on { border-color: var(--cu-accent, #6aa0ff); color: var(--cu-ink); }
.c2c-vloader__cands { display: flex; flex-direction: column; gap: 2px; font-size: 10px; color: var(--cu-ink-soft); }
.c2c-vloader__cands button { all: unset; cursor: pointer; padding: 1px 4px; border-radius: 3px;
  font-variant-numeric: tabular-nums; }
.c2c-vloader__cands button:hover, .c2c-vloader__cands button:focus-visible { background: var(--cu-raised); color: var(--cu-ink); }
.c2c-vloader--drop { outline: 2px dashed var(--cu-accent, #6aa0ff); outline-offset: -2px; }
`;
    document.head.appendChild(s);
}

let formatsPromise = null;
/** The extensions an upload may carry - what this ComfyUI's backends actually read (Rev 3 3.3). */
export function mediaFormats() {
    if (!formatsPromise) {
        formatsPromise = api.fetchApi("/c2c/media/formats")
            .then((r) => (r.ok ? r.json() : { extensions: [] }))
            .catch(() => ({ extensions: [] }));
    }
    return formatsPromise;
}

/** POST the files; resolves with the route's JSON ({files, subfolder, value, sequences}) or rejects in words. */
export function uploadMedia(files, { overwrite = false, onProgress = null } = {}) {
    return new Promise((resolve, reject) => {
        const fd = new FormData();
        fd.append("overwrite", overwrite ? "true" : "false");
        for (const f of files) {
            if (f.webkitRelativePath) fd.append("relpath", f.webkitRelativePath);   // a folder upload keeps its folder
            fd.append("file", f, f.name);
        }
        const xhr = new XMLHttpRequest();                     // fetch() has no upload progress
        xhr.open("POST", api.apiURL("/c2c/media/upload"));
        xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress?.(e.loaded / e.total); };
        xhr.onload = () => {
            let body = null;
            try { body = JSON.parse(xhr.responseText); } catch (_e) { /* not JSON */ }
            if (xhr.status >= 200 && xhr.status < 300 && body) resolve(body);
            else reject(new Error(body?.error || `Upload failed (HTTP ${xhr.status}).`));
        };
        xhr.onerror = () => reject(new Error("Upload failed: the server did not answer."));
        xhr.send(fd);
    });
}

function summarize(res) {
    const n = res.files?.length || 0;
    const by = {};
    for (const f of res.files || []) by[f.status] = (by[f.status] || 0) + 1;
    const extra = Object.entries(by).filter(([k]) => k !== "written").map(([k, v]) => `${v} ${k}`).join(", ");
    return `Uploaded ${n} file${n === 1 ? "" : "s"}${extra ? ` (${extra})` : ""} -> ${res.value}`;
}

/**
 * The bar above the preview. `ctx`: {node, ui, setWidget, widgetValue, refresh, fit}. Returns {el}.
 */
export function buildSourceBar(ctx) {
    ensureStyles();
    const { node, ui } = ctx;
    const el = document.createElement("div");
    el.className = "c2c-vloader__source";
    const progress = document.createElement("div");
    progress.className = "c2c-vloader__progress";
    progress.hidden = true;
    const fill = document.createElement("div");
    progress.appendChild(fill);

    const over = document.createElement("label");
    over.className = "c2c-vloader__over";
    over.title = "Replace files with the same name instead of keeping both (off: the same bytes are reused, other "
        + "bytes get a new name, a sequence a new folder).";
    const overBox = document.createElement("input");
    overBox.type = "checkbox";
    over.append(overBox, document.createTextNode("overwrite"));

    const picker = (folder) => {
        const input = document.createElement("input");
        input.type = "file";
        input.multiple = true;
        input.hidden = true;                                   // in the bar (not detached): the same element a test drives
        input.dataset.c2cUpload = folder ? "folder" : "files";
        if (folder) input.webkitdirectory = true;
        mediaFormats().then((f) => { if (!folder && f.extensions?.length) input.accept = f.extensions.join(","); });
        input.addEventListener("change", () => {
            const files = [...(input.files || [])];
            if (files.length) send(files);
        });
        return input;
    };

    async function send(files) {
        const fmts = (await mediaFormats()).extensions || [];
        const ok = fmts.length ? files.filter((f) => fmts.includes(("." + (f.name.split(".").pop() || "")).toLowerCase())) : files;
        const skipped = files.length - ok.length;
        if (!ok.length) {
            ui.status.setText(`None of these files is a format this ComfyUI reads (${fmts.join(" ")}).`, "danger");
            return;
        }
        progress.hidden = false;
        fill.style.width = "0%";
        ui.status.setText(`Uploading ${ok.length} file${ok.length === 1 ? "" : "s"}…`, "default");
        ctx.fit();
        try {
            const res = await uploadMedia(ok, {
                overwrite: overBox.checked,
                onProgress: (p) => { fill.style.width = `${Math.round(p * 100)}%`; },
            });
            const w = node.widgets?.find((x) => x.name === "video");
            if (w?.options?.values && !w.options.values.includes(res.value)) w.options.values.push(res.value);
            const note = summarize(res) + (skipped ? ` · ${skipped} unreadable file(s) left out` : "");
            ctx.note = note;                                   // shown again after the probe that follows
            ctx.setWidget(node, "location", "input folder");
            ctx.setWidget(node, "video", res.value);
            ui.status.setText(note, "ok");
        } catch (err) {
            ui.status.setText(err.message || "Upload failed.", "danger");
        } finally {
            progress.hidden = true;
            ctx.fit();
        }
    }

    const files = picker(false);
    const folder = picker(true);
    el.append(
        button("Upload files…", { onClick: () => { files.value = ""; files.click(); } }),
        button("Upload folder…", { onClick: () => { folder.value = ""; folder.click(); } }),
        over,
        button("↻", { onClick: () => ctx.refresh() }),
        progress,
        files,
        folder,
    );
    el.querySelectorAll("button")[0].title = "Videos, images, EXRs - several at once for a sequence. You can also drop "
        + "files on the node.";
    el.querySelectorAll("button")[1].title = "A folder of frames (kept as one folder in the input folder).";
    el.querySelectorAll("button")[2].title = "Read the source again (new frames, a changed file).";

    // drag-and-drop onto the node (classic canvas calls these; Nodes 2.0 forwards the drop to them too)
    const prevOver = node.onDragOver;
    node.onDragOver = function (e) {
        const files = !!e?.dataTransfer?.types && [...e.dataTransfer.types].includes("Files");
        ui.root.classList.toggle("c2c-vloader--drop", files);
        return files || !!prevOver?.call(this, e);
    };
    const prevDrop = node.onDragDrop;
    node.onDragDrop = function (e) {
        ui.root.classList.remove("c2c-vloader--drop");
        const list = [...(e?.dataTransfer?.files || [])];
        if (!list.length) return !!prevDrop?.call(this, e);
        send(list);
        return true;
    };
    ui.root.addEventListener("dragleave", () => ui.root.classList.remove("c2c-vloader--drop"));
    return { el, send };
}

const KIND = { video: "movie", image_sequence: "image sequence", exr_sequence: "EXR sequence", still: "single image",
               exr_still: "single EXR" };

/** "30000/1001" -> "29.97", "24/1" -> "24" */
function fpsText(f) {
    const [n, d] = String(f || "").split("/").map(Number);
    const v = d ? n / d : Number(f);
    return Number.isFinite(v) ? String(Math.round(v * 1000) / 1000) : String(f || "");
}

function chip(text, cls = "", title = "") {
    const c = document.createElement("span");
    c.className = "c2c-vloader__chip" + (cls ? " " + cls : "");
    c.textContent = text;
    if (title) c.title = title;
    return c;
}

/**
 * Source chips after the probe's media facts: what it is, its frame numbers, missing frames, layers (a layer chip
 * sets exr_layer), and the other sequences sharing the name (a click loads that one instead - never merged).
 */
export function renderSource(ui, media, ctx) {
    ensureStyles();
    ui.info.querySelectorAll(".c2c-vloader__chip--src, .c2c-vloader__cands").forEach((e) => e.remove());
    if (!media) return;
    const add = (c) => { c.classList.add("c2c-vloader__chip--src"); ui.info.appendChild(c); return c; };
    const seq = media.sequence;
    add(chip(KIND[media.kind] || media.kind));
    if (seq) add(chip(`${seq.pattern} · ${seq.first}–${seq.last}${seq.step > 1 ? ` x${seq.step}` : ""}`, "",
        `${seq.count} files in ${seq.directory}`));
    if (media.fps_from && media.kind !== "video") {
        add(chip(`${fpsText(media.fps)} fps (${media.fps_from})`, "", "0 in sequence_fps = the file's own rate, else 24"));
    }
    const tl = media.timeline || {};
    if (tl.missing_count) {
        add(chip(`${tl.missing_count} missing`, "c2c-vloader__chip--warn",
            `Missing frame numbers: ${(tl.missing || []).join(", ")}${tl.missing_count > (tl.missing || []).length ? " …" : ""}`));
    } else if (seq?.missing_count) {
        add(chip(`${seq.missing_count} missing in the folder`, "c2c-vloader__chip--warn",
            `Not in the selected range, or filled by missing_frames: ${(seq.missing || []).join(", ")}`));
    }
    if (media.data_layer) add(chip("raw data values", "", "A data layer: loaded as values, never colour-managed"));
    const layers = media.layers || [];
    if (layers.length > 1 || (layers.length === 1 && layers[0].key !== "rgba")) {
        for (const lay of layers) {
            const on = lay.key === media.layer;
            const c = add(chip(lay.key, "c2c-vloader__chip--pick" + (on ? " c2c-vloader__chip--on" : ""),
                `${lay.kind}${lay.data ? " (data)" : ""}: ${lay.channels.join(", ")}${lay.alpha ? " + " + lay.alpha : ""}`
                + (on ? " - shown in IMAGE" : " - click to load this layer")));
            c.addEventListener("click", () => ctx.setWidget(ctx.node, "exr_layer", lay.key));
        }
    }
    const cands = (media.candidates || []).filter((c) => !seq || c.pattern !== seq.pattern);
    if (cands.length) {
        const box = document.createElement("div");
        box.className = "c2c-vloader__cands";
        box.appendChild(document.createTextNode("Also in this folder (not merged):"));
        for (const c of cands) {
            const b = document.createElement("button");
            b.type = "button";
            b.textContent = `${c.pattern} · ${c.first}–${c.last} · ${c.count} f${c.padding ? ` · pad ${c.padding}` : ""}`;
            b.title = "Load this sequence instead";
            b.addEventListener("click", () => {
                const cur = String(ctx.widgetValue(ctx.node, ctx.widgetValue(ctx.node, "location") === "server path"
                    ? "path" : "video") || "");
                const dir = cur.includes("/") || cur.includes("\\") ? cur.replace(/[^/\\]*$/, "") : "";
                const name = ctx.widgetValue(ctx.node, "location") === "server path" ? "path" : "video";
                ctx.setWidget(ctx.node, name, dir + c.pattern);
            });
            box.appendChild(b);
        }
        ui.info.appendChild(box);
    }
    if (ctx.note) { ui.status.setText(ctx.note, "ok"); ctx.note = null; }     // the upload's result, once
    for (const w of media.warnings || []) ui.status.setText(w, "warn");
}
